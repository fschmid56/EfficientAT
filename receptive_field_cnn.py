import argparse
from models.mn.model import get_model
from helpers.utils import NAME_TO_WIDTH
from helpers.receptive_field import receptive_field_cnn


def calc_receptive_field(args):
    # model
    if args.model_width:
        # manually specified settings
        width = args.model_width
        model_name = "mn{}".format(str(width).replace(".", ""))
    else:
        # model width specified via model name
        model_name = args.model_name
        width = NAME_TO_WIDTH(model_name)
    model = get_model(width_mult=width, se_dims=args.se_dims, head_type=args.head_type, strides=args.strides)
    model.eval()

    rf_freq, rf_time = receptive_field_cnn(model, (1, 1, 128, 1000))
    print(f"Receptive field size of {model_name} with strides {args.strides}: Frequency: {rf_freq}, Time: {rf_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # model name decides, which pre-trained model is evaluated in terms of complexity
    parser.add_argument('--model_name', type=str, default='mn10_as')
    # alternatively, specify model configurations manually
    parser.add_argument('--model_width', type=float, default=None)
    parser.add_argument('--head_type', type=str, default='mlp')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--se_dims', type=str, default='c')

    args = parser.parse_args()
    calc_receptive_field(args)
