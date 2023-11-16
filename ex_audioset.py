import wandb
import numpy as np
import os
from torch import autocast
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
import pickle

from datasets.audioset import get_test_set, get_full_training_set, get_ft_weighted_sampler
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup

preds_url = \
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/passt_enemble_logits_mAP_495.npy"

fname_to_index_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/fname_to_index.pkl"


def train(args):
    # Train Models from scratch or ImageNet pre-trained on AudioSet
    # PaSST ensemble (https://github.com/kkoutini/PaSST) stored in 'resources/passt_enemble_logits_mAP_495.npy'
    # can be used as a teacher.

    # logging is done using wandb
    wandb.init(
        project="EfficientAudioTagging",
        notes="Training efficient audio tagging models on AudioSet using Knowledge Distillation.",
        tags=["AudioSet", "Audio Tagging", "Knowledge Disitillation"],
        config=args,
        name=args.experiment_name
    )

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         freqm=args.freqm,
                         timem=args.timem,
                         fmin=args.fmin,
                         fmax=args.fmax,
                         fmin_aug_range=args.fmin_aug_range,
                         fmax_aug_range=args.fmax_aug_range
                         )
    mel.to(device)
    # load prediction model
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         strides=args.strides, pretrain_final_temp=args.pretrain_final_temp)
    else:
        model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              strides=args.strides, head_type=args.head_type, se_dims=args.se_dims)
    model.to(device)

    # dataloader
    dl = DataLoader(dataset=get_full_training_set(resample_rate=args.resample_rate, roll=args.roll, wavmix=args.wavmix,
                                                  gain_augment=args.gain_augment),
                    sampler=get_ft_weighted_sampler(args.epoch_len),  # sampler important to balance classes
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)

    # evaluation loader
    eval_dl = DataLoader(dataset=get_test_set(resample_rate=args.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)

    if args.adamw:
        # optimizer & scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    else:
        # optimizer & scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)


    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    # prepare ingredients for knowledge distillation
    assert 0 <= args.kd_lambda <= 1, "Lambda for Knowledge Distillation must be between 0 and 1."
    distillation_loss = nn.BCEWithLogitsLoss(reduction="none")
    # load stored teacher predictions

    if not os.path.isfile(args.teacher_preds):
        # download file
        print("Download teacher predictions...")
        download_url_to_file(preds_url, args.teacher_preds)
    print(f"Load teacher predictions from file {args.teacher_preds}")
    teacher_preds = np.load(args.teacher_preds)
    teacher_preds = torch.from_numpy(teacher_preds).float()
    teacher_preds = torch.sigmoid(teacher_preds / args.temperature)
    teacher_preds.requires_grad = False

    if not os.path.isfile(args.fname_to_index):
        print("Download filename to teacher prediction index dictionary...")
        download_url_to_file(fname_to_index_url, args.fname_to_index)
    with open(args.fname_to_index, 'rb') as f:
        fname_to_index = pickle.load(f)

    name = None
    mAP, ROC, val_loss = float('NaN'), float('NaN'), float('NaN')

    for epoch in range(args.n_epochs):
        mel.train()
        model.train()
        train_stats = dict(train_loss=list(), label_loss=list(), distillation_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}/{}: mAP: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, args.n_epochs, mAP, val_loss))

        # in case of DyMN: update DyConv temperature
        if hasattr(model, "update_params"):
            model.update_params(epoch)

        for batch in pbar:
            x, f, y, i = batch
            bs = x.size(0)
            x, y = x.to(device), y.to(device)
            x = _mel_forward(x, mel)

            rn_indices, lam = None, None
            if args.mixup_alpha:
                rn_indices, lam = mixup(bs, args.mixup_alpha)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                y_mix = y * lam.reshape(bs, 1) + y[rn_indices] * (1. - lam.reshape(bs, 1))
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y_mix, reduction="none")
            else:
                y_hat, _ = model(x)
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")

            # hard label loss
            label_loss = samples_loss.mean()

            # distillation loss
            if args.kd_lambda > 0:
                # fetch the correct index in 'teacher_preds' for given filename
                # insert -1 for files not in fname_to_index (proportion of files successfully downloaded from
                # YouTube can vary for AudioSet)
                indices = torch.tensor(
                    [fname_to_index[fname] if fname in fname_to_index else -1 for fname in f], dtype=torch.int64
                )
                # get indices of files we could not find the teacher predictions for
                unknown_indices = indices == -1
                y_soft_teacher = teacher_preds[indices]
                y_soft_teacher = y_soft_teacher.to(y_hat.device).type_as(y_hat)

                if args.mixup_alpha:
                    soft_targets_loss = \
                        distillation_loss(y_hat, y_soft_teacher).mean(dim=1) * lam.reshape(bs) + \
                        distillation_loss(y_hat, y_soft_teacher[rn_indices]).mean(dim=1) \
                        * (1. - lam.reshape(bs))
                else:
                    soft_targets_loss = distillation_loss(y_hat, y_soft_teacher)

                # zero out loss for samples we don't have teacher predictions for
                soft_targets_loss[unknown_indices] = soft_targets_loss[unknown_indices] * 0
                soft_targets_loss = soft_targets_loss.mean()

                # weighting losses
                label_loss = args.kd_lambda * label_loss
                soft_targets_loss = (1 - args.kd_lambda) * soft_targets_loss
            else:
                soft_targets_loss = torch.tensor(0., device=label_loss.device, dtype=label_loss.dtype)

            # total loss is sum of lambda-weighted label and distillation loss
            loss = label_loss + soft_targets_loss

            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())
            train_stats['label_loss'].append(label_loss.detach().cpu().numpy())
            train_stats['distillation_loss'].append(soft_targets_loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Update learning rate
        scheduler.step()

        # evaluate
        mAP, ROC, val_loss = _test(model, mel, eval_dl, device)

        # log train and validation statistics
        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "label_loss": np.mean(train_stats['label_loss']),
                   "distillation_loss": np.mean(train_stats['distillation_loss']),
                   "learning_rate": scheduler.get_last_lr()[0],
                   "mAP": mAP,
                   "ROC": ROC,
                   "val_loss": val_loss
                   })

        # remove previous model (we try to not flood your hard disk) and save latest model
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = f"mn{str(width).replace('.', '')}_as_epoch_{epoch}_mAP_{int(round(mAP*1000))}.pt"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.binary_cross_entropy_with_logits(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)
    return mAP.mean(), ROC.mean(), losses.mean()


def evaluate(args):
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # load pre-trained model
    if len(args.ensemble) > 0:
        print(f"Running AudioSet evaluation for models '{args.ensemble}' on device '{device}'")
        model = get_ensemble_model(args.ensemble)
    else:
        print(f"Running AudioSet evaluation for model '{model_name}' on device '{device}'")
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                             strides=args.strides)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=args.strides, head_type=args.head_type)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         fmin=args.fmin,
                         fmax=args.fmax
                         )
    mel.to(device)
    mel.eval()

    dl = DataLoader(dataset=get_test_set(resample_rate=args.resample_rate),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)

    targets = []
    outputs = []
    for batch in tqdm(dl):
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        # our models are trained in half precision mode (torch.float16)
        # run on cuda with torch.float16 to get the best performance
        # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
        with autocast(device_type=device.type) if args.cuda else nullcontext():
            with torch.no_grad():
                x = _mel_forward(x, mel)
                y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)

    if len(args.ensemble) > 0:
        print(f"Results on AudioSet test split for loaded models: {args.ensemble}")
    else:
        print(f"Results on AudioSet test split for loaded model: {model_name}")
    print("  mAP: {:.3f}".format(mAP.mean()))
    print("  ROC: {:.3f}".format(ROC.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=12)

    # evaluation
    # if ensemble is set, 'model_name' is not used
    parser.add_argument('--ensemble', nargs='+', default=[])
    parser.add_argument('--model_name', type=str, default="mn10_as")  # used also for training

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrain_final_temp', type=float, default=30.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--epoch_len', type=int, default=100000)
    parser.add_argument('--roll', action='store_true', default=False)
    parser.add_argument('--wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=0)

    # optimizer
    parser.add_argument('--adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0)
    # lr schedule
    parser.add_argument('--max_lr', type=float, default=0.0008)
    parser.add_argument('--warm_up_len', type=int, default=8)
    parser.add_argument('--ramp_down_start', type=int, default=80)
    parser.add_argument('--ramp_down_len', type=int, default=95)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # knowledge distillation
    parser.add_argument('--teacher_preds', type=str,
                        default=os.path.join("resources", "passt_enemble_logits_mAP_495.npy"))
    parser.add_argument('--fname_to_index', type=str,
                        default=os.path.join("resources", "fname_to_index.pkl"))
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--kd_lambda', type=float, default=0.1)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()
    if args.train:
        train(args)
    else:
        evaluate(args)
