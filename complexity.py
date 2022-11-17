import argparse

from helpers.flop_count import count_macs, count_macs_transformer
from models.MobileNetV3 import get_model
from helpers.utils import NAME_TO_WIDTH


def calc_complexity(args):
    # MobileNet
    model_name = args.model_name
    model = get_model(width_mult=NAME_TO_WIDTH[model_name])
    model.eval()
    # result for creating spectrogram of a 10-seconds audio clip is of shape 1, 1, 128 (n_mels), 1000 (time resolution)
    total_macs = count_macs(model, (1, 1, 128, 1000))
    total_params = sum(p.numel() for p in model.parameters())

    print("Model '{}' has {:.2f} million parameters and inference of a single 10-seconds audio clip requires "
          "{:.2f} billion multiply-accumulate operations.".format(model_name, total_params/10**6, total_macs/10**9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn10_as')
    args = parser.parse_args()
    calc_complexity(args)
