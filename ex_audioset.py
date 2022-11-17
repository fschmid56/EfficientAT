import numpy as np
from torch import autocast
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
from contextlib import nullcontext

from datasets.audioset import get_test_set, get_training_set
from models.MobileNetV3 import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH


def train(args):
    # This will follow in one of the next releases.
    # It will show how the knowledge distillation was set up and how the
    # PaSST ensemble (https://github.com/kkoutini/PaSST) stored in 'resources/passt_enemble_logits_mAP_495.npy'
    # can be used as a teacher.
    pass


def evaluate(args):
    model_name = args.model
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # load pre-trained model
    model = get_mobilenet(width_mult=NAME_TO_WIDTH[model_name], pretrained_name=model_name)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=args.n_mels, sr=args.sample_rate, win_length=args.window_size,
                         hopsize=args.hop_size)
    mel.to(device)
    mel.eval()

    dl = DataLoader(dataset=get_test_set(),
                    worker_init_fn=worker_init_fn,
                    num_workers=12,
                    batch_size=64)

    print(f"Running AudioSet evaluation for model '{model_name}' on device '{device}'")
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
                x = mel(x.squeeze())
            x = x.unsqueeze(1)
            with torch.no_grad():
                y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)

    print(f"Results on AudioSet test split for loaded model: {model_name}")
    print(f"  mAP: {mAP.mean()}")
    print(f"  ROC: {ROC.mean()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    parser.add_argument('--model', type=str, default='mn10_as')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)
    args = parser.parse_args()
    if args.train:
        train(args)
    else:
        evaluate(args)
