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
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

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


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # model to preprocess waveform to mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                         sr=config.resample_rate,
                         win_length=config.window_size,
                         hopsize=config.hop_size,
                         n_fft=config.n_fft,
                         freqm=config.freqm,
                         timem=config.timem,
                         fmin=config.fmin,
                         fmax=config.fmax,
                         fmin_aug_range=config.fmin_aug_range,
                         fmax_aug_range=config.fmax_aug_range
                         )

        # load prediction model
        model_name = config.model_name
        pretrained_name = model_name if config.pretrained else None
        width = NAME_TO_WIDTH(model_name) if model_name and config.pretrained else config.model_width
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                             strides=config.strides, pretrain_final_temp=config.pretrain_final_temp)
        else:
            model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                                  strides=config.strides, head_type=config.head_type, se_dims=config.se_dims)
        self.model = model

        # prepare ingredients for knowledge distillation
        assert 0 <= config.kd_lambda <= 1, "Lambda for Knowledge Distillation must be between 0 and 1."
        self.distillation_loss = nn.BCEWithLogitsLoss(reduction="none")

        # load stored teacher predictions
        if not os.path.isfile(config.teacher_preds):
            # download file
            print("Download teacher predictions...")
            download_url_to_file(preds_url, config.teacher_preds)
        print(f"Load teacher predictions from file {config.teacher_preds}")
        teacher_preds = np.load(config.teacher_preds)
        teacher_preds = torch.from_numpy(teacher_preds).float()
        teacher_preds = torch.sigmoid(teacher_preds / config.temperature)
        teacher_preds.requires_grad = False
        self.teacher_preds = teacher_preds

        if not os.path.isfile(config.fname_to_index):
            print("Download filename to teacher prediction index dictionary...")
            download_url_to_file(fname_to_index_url, config.fname_to_index)
        with open(config.fname_to_index, 'rb') as f:
            fname_to_index = pickle.load(f)
        self.fname_to_index = fname_to_index

        self.distributed_mode = config.num_devices > 1
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        if self.config.adamw:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.max_lr,
                                          weight_decay=self.config.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.max_lr,
                                         weight_decay=self.config.weight_decay)

        # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def on_train_epoch_start(self):
        # in case of DyMN: update DyConv temperature
        if hasattr(self.model, "update_params"):
            self.model.update_params(self.current_epoch)

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """
        x, f, y, i = train_batch
        bs = x.size(0)
        x = self.mel_forward(x)

        rn_indices, lam = None, None
        if self.config.mixup_alpha:
            rn_indices, lam = mixup(bs, self.config.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(bs, 1, 1, 1) + \
                x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
            y_hat, _ = self.model(x)
            y_mix = y * lam.reshape(bs, 1) + y[rn_indices] * (1. - lam.reshape(bs, 1))
            samples_loss = F.binary_cross_entropy_with_logits(y_hat, y_mix, reduction="none")
        else:
            y_hat, _ = self.model(x)
            samples_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")

        # hard label loss
        label_loss = samples_loss.mean()

        # distillation loss
        if self.config.kd_lambda > 0:
            # fetch the correct index in 'teacher_preds' for given filename
            # insert -1 for files not in fname_to_index (proportion of files successfully downloaded from
            # YouTube can vary for AudioSet)
            indices = torch.tensor(
                [self.fname_to_index[fname] if fname in self.fname_to_index else -1 for fname in f], dtype=torch.int64
            )
            # get indices of files we could not find the teacher predictions for
            unknown_indices = indices == -1
            y_soft_teacher = self.teacher_preds[indices]
            y_soft_teacher = y_soft_teacher.to(y_hat.device).type_as(y_hat)

            if self.config.mixup_alpha:
                soft_targets_loss = \
                    self.distillation_loss(y_hat, y_soft_teacher).mean(dim=1) * lam.reshape(bs) + \
                    self.distillation_loss(y_hat, y_soft_teacher[rn_indices]).mean(dim=1) \
                    * (1. - lam.reshape(bs))
            else:
                soft_targets_loss = distillation_loss(y_hat, y_soft_teacher)

            # zero out loss for samples we don't have teacher predictions for
            soft_targets_loss[unknown_indices] = soft_targets_loss[unknown_indices] * 0
            soft_targets_loss = soft_targets_loss.mean()

            # weighting losses
            label_loss = self.config.kd_lambda * label_loss
            soft_targets_loss = (1 - self.config.kd_lambda) * soft_targets_loss
        else:
            soft_targets_loss = torch.tensor(0., device=label_loss.device, dtype=label_loss.dtype)

        # total loss is sum of lambda-weighted label and distillation loss
        loss = label_loss + soft_targets_loss

        results = {"loss": loss.detach().cpu(), "label_loss": label_loss.detach().cpu(),
                   "kd_loss": soft_targets_loss.detach().cpu()}
        self.training_step_outputs.append(results)
        return loss

    def on_train_epoch_end(self):
        """
        :return: a dict containing the metrics you want to log to Weights and Biases
        """
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_label_loss = torch.stack([x['label_loss'] for x in self.training_step_outputs]).mean()
        avg_kd_loss = torch.stack([x['kd_loss'] for x in self.training_step_outputs]).mean()
        self.log_dict({'train/loss': torch.as_tensor(avg_loss).cuda(),
                       'train/label_loss': torch.as_tensor(avg_label_loss).cuda(),
                       'train/kd_loss': torch.as_tensor(avg_kd_loss).cuda()
                       }, sync_dist=True)

        self.training_step_outputs.clear()

    def validation_step(self, val_batch, batch_idx):
        x, _, y = val_batch
        x = self.mel_forward(x)
        y_hat, _ = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        preds = torch.sigmoid(y_hat)
        results = {'val_loss': loss, "preds": preds, "targets": y}
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs])
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0)

        all_preds = self.all_gather(preds).reshape(-1, preds.shape[-1]).cpu().float().numpy()
        all_targets = self.all_gather(targets).reshape(-1, targets.shape[-1]).cpu().float().numpy()
        all_loss = self.all_gather(loss).reshape(-1,)

        try:
            average_precision = metrics.average_precision_score(
                all_targets, all_preds, average=None)
        except ValueError:
            average_precision = np.array([np.nan] * 527)
        try:
            roc = metrics.roc_auc_score(all_targets, all_preds, average=None)
        except ValueError:
            roc = np.array([np.nan] * 527)
        logs = {'val/loss': torch.as_tensor(all_loss).mean().cuda(),
                'val/ap': torch.as_tensor(average_precision).mean().cuda(),
                'val/roc': torch.as_tensor(roc).mean().cuda()
                }
        self.log_dict(logs, sync_dist=False)
        self.validation_step_outputs.clear()


def train(config):
    # Train Models from scratch or ImageNet pre-trained on AudioSet
    # PaSST ensemble (https://github.com/kkoutini/PaSST) stored in 'resources/passt_enemble_logits_mAP_495.npy'
    # can be used as a teacher.

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="EfficientAudioTagging",
        notes="Training efficient audio tagging models on AudioSet using Knowledge Distillation.",
        tags=["AudioSet", "Audio Tagging", "Knowledge Disitillation"],
        config=config,
        name=config.experiment_name
    )

    train_dl = DataLoader(dataset=get_full_training_set(resample_rate=config.resample_rate,
                                                        roll=config.roll,
                                                        wavmix=config.wavmix,
                                                        gain_augment=config.gain_augment),
                          sampler=get_ft_weighted_sampler(config.epoch_len),  # sampler important to balance classes
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size)

    # eval dataloader
    eval_dl = DataLoader(dataset=get_test_set(resample_rate=config.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)

    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         callbacks=[lr_monitor])

    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, eval_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet")
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_devices', type=int, default=4)

    # evaluation
    # if ensemble is set, 'model_name' is not used
    parser.add_argument('--ensemble', nargs='+', default=[])
    parser.add_argument('--model_name', type=str, default="mn10_as")  # used also for training
    parser.add_argument('--cuda', action='store_true', default=False)

    # training
    parser.add_argument('--precision', type=int, default=16)
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
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # lr schedule
    parser.add_argument('--max_lr', type=float, default=0.003)
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
    train(args)
