import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F

from datasets.fsd50k import get_eval_set, get_valid_set, get_training_set
from models.MobileNetV3 import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup


def train(args):
    # Train Models on FSD50K

    # logging is done using wandb
    wandb.init(
        project="FSD50K",
        notes="Fine-tune Models on FSD50K.",
        tags=["FSDK50K", "Audio Tagging"],
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
    pretrained_name = args.pretrained_name
    if pretrained_name:
        model_width = NAME_TO_WIDTH(pretrained_name)
        model = get_mobilenet(width_mult=model_width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims, num_classes=200)
    else:
        model_width = args.model_width
        model = get_mobilenet(width_mult=model_width, head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=200)
    model.to(device)

    # dataloader
    dl = DataLoader(dataset=get_training_set(resample_rate=args.resample_rate, roll=args.roll,
                                             wavmix=args.wavmix, gain_augment=args.gain_augment),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

    # evaluation loader
    valid_dl = DataLoader(dataset=get_valid_set(resample_rate=args.resample_rate,
                                                variable_eval=args.variable_eval_length),
                          worker_init_fn=worker_init_fn,
                          num_workers=args.num_workers,
                          batch_size=1 if args.variable_eval_length else args.batch_size)

    # optimizer & scheduler
    lr = args.lr
    features_lr = args.features_lr if args.features_lr else lr
    classifier_lr = args.classifier_lr if args.classifier_lr else lr
    last_layer_lr = args.last_layer_lr if args.last_layer_lr else classifier_lr

    assert args.classifier_lr is None or args.last_layer_lr is None, "Either specify separate learning rate for " \
                                                                     "last layer or classifier, not both."

    optimizer = torch.optim.Adam([{'params': model.features.parameters(), 'lr': features_lr},
                                  {'params': model.classifier[:5].parameters(), 'lr': classifier_lr},
                                  {'params': model.classifier[5].parameters(), 'lr': last_layer_lr}
                                  ],
                                 lr=args.lr, weight_decay=args.weight_decay)
    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    name = None
    mAP, ROC, val_loss = float('NaN'), float('NaN'), float('NaN')

    for epoch in range(args.n_epochs):
        mel.train()
        model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}/{}: mAP: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, args.n_epochs, mAP, val_loss))
        for batch in pbar:
            x, f, y = batch
            bs = x.size(0)
            x, y = x.to(device), y.to(device)
            x = _mel_forward(x, mel)

            if args.mixup_alpha:
                rn_indices, lam = mixup(bs, args.mixup_alpha)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                y_mix = y * lam.reshape(bs, 1) + y[rn_indices] * (1. - lam.reshape(bs, 1))
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y_mix, reduction="none")
            else:
                x = model.features(x)
                y_hat = model.classifier(x)
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")

            # loss
            loss = samples_loss.mean()

            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Update learning rate
        scheduler.step()

        # evaluate
        mAP, ROC, val_loss = _test(model, mel, valid_dl, device)

        # log train and validation statistics
        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "learning_rate": scheduler.get_last_lr()[0],
                   "mAP": mAP,
                   "ROC": ROC,
                   "val_loss": val_loss
                   })

        # remove previous model (we try to not flood your hard disk) and save latest model
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = f"mn{str(model_width).replace('.', '')}_fsd50k_epoch_{epoch}_mAP_{int(round(mAP*1000))}.pt"
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
    model_name = args.pretrained_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # load pre-trained model
    model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name, num_classes=200)
    model.to(device)
    model.eval()

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
    mel.eval()

    dl = DataLoader(dataset=get_eval_set(resample_rate=args.resample_rate,
                                         variable_eval=args.variable_eval_length),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=1 if args.variable_eval_length else args.batch_size)

    print(f"Running FSD50K evaluation for model '{model_name}' on device '{device}'")
    targets = []
    outputs = []
    for batch in tqdm(dl):
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)

    print(f"Results on AudioSet test split for loaded model: {model_name}")
    print("  mAP: {:.3f}".format(mAP.mean()))
    print("  ROC: {:.3f}".format(ROC.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="FSD50K")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12)

    # validation & evaluation
    # required setting validation and evaluation batch size to 1
    parser.add_argument('--variable_eval_length', action='store_true', default=False)

    # training
    parser.add_argument('--pretrained_name', type=str, default=None)
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--mixup_alpha', type=float, default=0)
    parser.add_argument('--roll', action='store_true', default=False)
    parser.add_argument('--wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=0)
    parser.add_argument('--weight_decay', type=int, default=0.0001)
    # lr schedule
    parser.add_argument('--lr', type=float, default=5e-5)
    # individual learning rates possible for classifier, features or last layer
    parser.add_argument('--classifier_lr', type=float, default=None)
    parser.add_argument('--last_layer_lr', type=float, default=None)
    parser.add_argument('--features_lr', type=float, default=None)
    parser.add_argument('--warm_up_len', type=int, default=0)
    parser.add_argument('--ramp_down_start', type=int, default=6)
    parser.add_argument('--ramp_down_len', type=int, default=9)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

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
