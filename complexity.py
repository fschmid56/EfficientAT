import argparse
import torch

from helpers.flop_count import count_macs, count_macs_transformer
from helpers.peak_memory import peak_memory_mnv3, peak_memory_cnn
from models.mn.model import get_model
from helpers.utils import NAME_TO_WIDTH
from models.preprocess import AugmentMelSTFT


def calc_complexity(args):
    # mel
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft
                         )

    # model
    if args.model_width:
        # manually specified settings
        width = args.model_width
        model_name = "mn{}".format(str(width).replace(".", ""))
    else:
        # model width specified via model name
        model_name = args.model_name
        width = NAME_TO_WIDTH(model_name)
    model = get_model(width_mult=width, se_dims=args.se_dims, head_type=args.head_type)
    model.eval()

    # waveform
    waveform = torch.zeros((1, args.resample_rate * 10))  # 10 seconds waveform
    spectrogram = mel(waveform)
    # squeeze in channel dimension
    spectrogram = spectrogram.unsqueeze(1)
    if args.complexity_type == "computation":
        # use size of spectrogram to calculate multiply-accumulate operations
        total_macs = count_macs(model, spectrogram.size())
        total_params = sum(p.numel() for p in model.parameters())
        print("Model '{}' has {:.2f} million parameters and inference of a single 10-seconds audio clip requires "
              "{:.2f} billion multiply-accumulate operations.".format(model_name, total_params/10**6, total_macs/10**9))
    elif args.complexity_type == "memory":
        if args.memory_efficient_inference:
            peak_mem = peak_memory_mnv3(model, spectrogram.size(), args.bits_per_elem)
            print("Model '{}' inference (memory efficient) of a single 10-seconds audio clip "
                  "has a peak memory of {:.2f} kB."
                  .format(model_name, peak_mem))
        else:
            peak_mem = peak_memory_cnn(model, spectrogram.size(), args.bits_per_elem)
            print("Model '{}' inference of a single 10-seconds audio clip has a peak memory of {:.2f} kB."
                  .format(model_name, peak_mem))
    else:
        raise NotImplementedError(f"Unknown complexity type: {args.complexity_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # either computation or memory complexity
    parser.add_argument('--complexity_type', type=str, default='computation')
    # for memory complexity
    parser.add_argument('--memory_efficient_inference', action='store_true', default=False)

    # model name decides, which pre-trained model is evaluated in terms of complexity
    parser.add_argument('--model_name', type=str, default='mn10_as')
    # alternatively, specify model configurations manually
    parser.add_argument('--model_width', type=float, default=None)
    parser.add_argument('--se_dims', type=str, default='c')
    parser.add_argument('--head_type', type=str, default='mlp')

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)

    # memory
    parser.add_argument('--bits_per_elem', type=int, default=16)

    args = parser.parse_args()
    calc_complexity(args)
