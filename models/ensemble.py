import torch.nn as nn

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from helpers.utils import NAME_TO_WIDTH


class EnsemblerModel(nn.Module):
    def __init__(self, models):
        super(EnsemblerModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        all_out = None
        for m in self.models:
            out, _ = m(x)
            if all_out is None:
                all_out = out
            else:
                all_out = out + all_out
        all_out = all_out / len(self.models)
        return all_out, all_out


def get_ensemble_model(model_names):
    models = []
    for model_name in model_names:
        if model_name.startswith("dymn"):
            model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
        else:
            model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
        models.append(model)
    return EnsemblerModel(models)
