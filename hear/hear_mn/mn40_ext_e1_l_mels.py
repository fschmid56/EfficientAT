import torch

from .models.MobileNetV3 import get_model
from .models.preprocess import AugmentMelSTFT
from .hear_wrapper import MNHearWrapper
from .helpers.utils import NAME_TO_WIDTH


def load_model(model_file_path="", model_name="mn40_as_ext", mode="all", add_averaged_mels=True):
    model = get_basic_model(model_name=model_name, mode=mode, add_averaged_mels=add_averaged_mels)
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_scene_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    model.eval()
    with torch.no_grad():
        return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
    model: Loaded Model.
    Returns:
    embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
    timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
    """
    model.eval()
    with torch.no_grad():
        return model.get_timestamp_embeddings(audio)


def get_basic_model(model_name="mn40_as_ext", **kwargs):
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024)
    net = get_model(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
    model = MNHearWrapper(mel=mel, net=net, scene_embedding_size=4495, timestamp_embedding_size=4495, **kwargs)
    return model
