# Efficient Pre-Trained CNNs for Audio Pattern Recognition

In this repository, we publish the pre-trained models and the code described in the papers:
* [Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation](https://arxiv.org/pdf/2211.04772.pdf). The paper has been presented in 
[ICASSP 2023](https://2023.ieeeicassp.org/) and is published in IEEE 
([published version](https://ieeexplore.ieee.org/abstract/document/10096110?casa_token=_FKutWF2kxIAAAAA:vtUj5_FKRHVxfWIs0nU4-GqW3jDkj6twAPaCxQrdV81AeiDcINsQU_zCK-iZZbJAHJXTRZZCkm3z)). 
* [Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio
Models](https://arxiv.org/pdf/2310.15648.pdf). Submitted to IEEE/ACM TASLP. **Pre-trained Models included, experiments on downstream tasks will be updated!**

The models in this repository are especially suited to you if you are looking for pre-trained audio pattern recognition models that are able to:
* achieve **competitive audio tagging performance on resource constrained platforms**
* reach **high performance on downstream tasks with a simple fine-tuning pipeline**
* extract **high-quality general purpose audio representations**

Pre-training Audio Pattern Recognition models by large-scale, general-purpose Audio Tagging is dominated by Transformers (PaSST [1], AST [2], HTS-AT [3], BEATs [16]) achieving the highest 
single-model mean average precisions (mAP) on AudioSet [4]. However, Transformers are complex
models and scale quadratically with respect to the sequence length, making them slow for inference.
CNNs scale linearly with respect to the sequence length and are easy to scale to given resource constraints. 
However, CNNs (e.g. PANNs [5], ERANN [6], PSLA [7]) have fallen short on Transformers in terms of Audio Tagging performance.

**We bring together the best of both worlds by training efficient CNNs of different complexities using Knowledge Distillation 
from Transformers**. The Figures below show the performance-complexity trade-off for existing models trained on AudioSet. The proposed MNs are described in [in this work](https://arxiv.org/pdf/2211.04772.pdf)
published at ICASSP 2023 and the DyMNs are introduced in our most recent [work](https://arxiv.org/pdf/2310.15648.pdf) submitted to TASLP.
The plots below are created using the model profiler included in Microsoft's [DeepSpeed framework](https://www.microsoft.com/en-us/research/project/deepspeed/).

![Model Performance vs. Model Size](/images/model_params_TASLP.png)

![Model Performance vs. Computational Complexity](/images/model_macs_TASLP.png)

Based on a reviewer request for the published ICASSP paper, we add the inference memory complexity of our pre-trained MNs.
We calculate the analytical peak memory (memory requirement of input + output activations) as in [14].
We also take into account memory-efficient inference in MobileNets as described in [15].

The plot below compares the trend in peak memory requirement between different CNNs. We use the file [peak_memory.py](helpers/peak_memory.py)
to determine the peak memory. The memory requirement is calculated assuming a 10 seconds audio snippet and fp16 representation
for all models.

![Model Performance vs. Memory Complexity](/images/mem_comp.png)


The next milestones are:
* Add the fine-tuning pipeline used in the [DyMN](https://arxiv.org/pdf/2310.15648.pdf) paper submitted to TASLP 
* Wrap this repository in an installable python package
* Use pytorch lightening to enable distributed training and training with fp16 

The final repository should have similar capabilities as the [PANNs codebase](https://github.com/qiuqiangkong/audioset_tagging_cnn)
with two main advantages:
* Pre-trained models of **lower computational and parameter complexity** due to the efficient CNN architectures
* **Higher performance** due to Knowledge Distillation from Transformers and optimized models

This codebase is inspired by the [PaSST](https://github.com/kkoutini/PaSST) and 
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) repositories, and the [pytorch implementation of
MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html).

## Environment

The codebase is developed with *Python 3.10.8*. After creating an environment install the requirements as
follows:

```
pip install -r requirements.txt
```

Also make sure you have FFmpeg <v4.4 installed.

## Pre-Trained Models



**Pre-trained models are available in the Github Releases and are automatically downloaded from there.** 
Loading the pre-trained models is as easy as running the following code pieces:

Pre-trained MobileNet:

```
from models.mn.model import get_model as get_mn
model = get_mn(pretrained_name="mn10_as")
```

Pre-trained Dynamic MobileNet:

```
from models.dymn.model import get_model as get_dymn
model = get_dymn(pretrained_name="dymn10_as")
```

The Table shows a selection of models contained in this repository. The naming convention for our models is 
**<model\>\<width_mult\>\_\<dataset\>**. In this sense, *mn10_as* defines a MobileNetV3 with parameter *width_mult=1.0*, pre-trained on 
AudioSet. *dymn* is the prefix for a dynamic MobileNet.

All models available are pre-trained on ImageNet [9] by default (otherwise denoted as 'no_im_pre'), followed by training on AudioSet [4]. Some results appear slightly better than those reported in the
papers. We provide the best models in this repository while the paper is showing averages over multiple runs.

| Model Name       | Config                                             | Params (Millions) | MACs (Billions) | Performance (mAP) |
|------------------|----------------------------------------------------|-------------------|-----------------|-------------------|
| dymn04_as        | width_mult=0.4                                     | 1.97              | 0.12            | 45.0              |
| dymn10_as        | width_mult=1.0                                     | 10.57             | 0.58            | 47.7              |
| dymn20_as        | width_mult=2.0                                     | 40.02             | 2.2             | 49.1              |
| mn04_as          | width_mult=0.4                                     | 0.983             | 0.11            | 43.2              |
| mn05_as          | width_mult=0.5                                     | 1.43              | 0.16            | 44.3              |
| mn10_as          | width_mult=1.0                                     | 4.88              | 0.54            | 47.1              |
| mn20_as          | width_mult=2.0                                     | 17.91             | 2.06            | 47.8              |
| mn30_as          | width_mult=3.0                                     | 39.09             | 4.55            | 48.2              |
| mn40_as          | width_mult=4.0                                     | 68.43             | 8.03            | 48.4              |
| mn40_as_ext      | width_mult=4.0,<br/>extended training (300 epochs) | 68.43             | 8.03            | 48.7              |
| mn40_as_no_im_pre      | width_mult=4.0, no ImageNet pre-training           | 68.43             | 8.03            | 48.3              |
| mn10_as_hop_15   | width_mult=1.0                                     | 4.88              | 0.36            | 46.3              |
| mn10_as_hop_20   | width_mult=1.0                                     | 4.88              | 0.27            | 45.6              |
| mn10_as_hop_25   | width_mult=1.0                                     | 4.88              | 0.22            | 44.7              |
| mn10_as_mels_40  | width_mult=1.0                                     | 4.88              | 0.21            | 45.3              |
| mn10_as_mels_64  | width_mult=1.0                                     | 4.88              | 0.27            | 46.1              |
| mn10_as_mels_256 | width_mult=1.0                                     | 4.88              | 1.08            | 47.4              |
| MN Ensemble         | width_mult=4.0, 9 Models                           | 615.87            | 72.27           | 49.8              |

MN Ensemble denotes an ensemble of 9 different mn40 models (3x mn40_as,
3x mn40_as_ext, 3x mn40_as_no_im_pre).

Note that computational complexity strongly depends on the resolution of the spectrograms. Our default is 128 mel bands and a hop size of 10 ms.

## Inference

You can use the pre-trained models for inference on an audio file using the 
[inference.py](inference.py) script.  

For example, use **dymn10_as** to detect acoustic events at a metro station in paris:

```
python inference.py --cuda --model_name=dymn10_as --audio_path="resources/metro_station-paris.wav"
```

This will result in the following output showing the 10 events detected with the highest probability:

```
************* Acoustic Event Detected: *****************
Train: 0.747
Subway, metro, underground: 0.599
Rail transport: 0.493
Railroad car, train wagon: 0.445
Vehicle: 0.360
Clickety-clack: 0.105
Speech: 0.053
Sliding door: 0.036
Outside, urban or manmade: 0.035
Music: 0.017
********************************************************
```

You can also use an ensemble for perform inference, e.g.:

```
python inference.py --ensemble dymn20_as mn40_as_ext mn40_as --cuda --audio_path=resources/metro_station-paris.wav
```


**Important:** All models are trained with half precision (float16). If you run float32 inference on cpu,
you might notice a slight performance degradation.

## Quality of extracted Audio Embeddings 

As shown in the paper [Low-Complexity Audio Embeddings Extractors](https://arxiv.org/pdf/2303.01879.pdf) (published at [EUSIPCO 2023](http://eusipco2023.org/)),
MNs are excellent at extracting high-quality audio embeddings. Checkout the repository [EfficientAT_HEAR](https://github.com/fschmid56/EfficientAT_HEAR)
for further details and the results on the [HEAR Benchmark](https://hearbenchmark.com/).

## Train and Evaluate on AudioSet

The training and evaluation procedures are simplified as much as possible. The most difficult part is to get AudioSet [4]
itself as it has a total size of around 1.1 TB and it must be downloaded from YouTube. Follow the instructions in 
the [PaSST](https://github.com/kkoutini/PaSST/tree/main/audioset) repository to get AudioSet in the format we need
to run the code in this repository. You should end up with three files:
* ```balanced_train_segmenets_mp3.hdf```
* ```unbalanced_train_segmenets_mp3.hdf```
* ```eval_segmenets_mp3.hdf```

Specify the folder containing the three files above in ```dataset_dir``` in the [dataset file](datasets/audioset.py).

Training and evaluation on AudioSet is implemented in the file [ex_audioset.py](ex_audioset.py).
#### Evaluation

To evaluate a model on the AudioSet evaluation data, run the following command:

```
python ex_audioset.py --cuda --model_name="dymn10_as"
```

Which will result in the following output:

```
Results on AudioSet test split for loaded model: dymn10_as
  mAP: 0.478
  ROC: 0.981
```

#### Training

Logging is done using [Weights & Biases](https://wandb.ai/site). Create a free account to log your experiments. During training
the latest model will be saved to the directory [wandb](wandb).

To train a MobileNet (pre-trained on ImageNet) on AudioSet, you can run, for example, the following command:
```
python ex_audioset.py --cuda --train --pretrained --model_name=mn10_im --batch_size=60 --max_lr=0.0004
```

Checkout the results of this example configuration [here](https://api.wandb.ai/links/florians/xbi9ijie).

To train a tiny model (```model_width=0.1```) with Squeeze-and-Excitation [10] on the frequency dimension and a fully convolutional
classification head, run the following command:

```
python ex_audioset.py --cuda --train --batch_size=120 --model_width=0.1 --head_type=fully_convolutional --se_dims=f
```

Checkout the results of this example configuration [here](https://api.wandb.ai/links/florians/k6e7o8qh).

To train a DyMN, pre-trained on ImageNet, run the following command:

```
python ex_audioset.py --cuda --train --pretrained --model_name=dymn10_im --batch_size=120 --max_lr=0.001 --pretrain_final_temp=30
```

Checkout the results of this example configuration [here](https://api.wandb.ai/links/florians/xu2v0on7).

To train a DyMN, pre-trained on ImageNet, using Adamw optimizer and a weight decay, run the following command:

```
python ex_audioset.py --cuda --train --pretrained --model_name=dymn10_im --batch_size=120 --max_lr=0.001 --pretrain_final_temp=30 --adamw --weight_decay=0.0001
```

Checkout the results of this example configuration [here](https://api.wandb.ai/links/florians/00pijva0).


A similar performance can be achieved by scaling down batch size and learning rate proportionally.  

For instance, the following command runs on a *NVIDIA GeForce RTX 2080 Ti* with 11 GB of memory.

```
python ex_audioset.py --cuda --train --pretrained --model_name=dymn10_im --batch_size=48 --max_lr=0.0004 --pretrain_final_temp=30
```

Checkout the results of this example configuration [here](https://api.wandb.ai/links/florians/532z11rr).

## Fine-tune on FSD50K [12]

Follow the instructions in the [PaSST](https://github.com/kkoutini/PaSST/tree/main/fsd50k) repository to get the FSD50K dataset.

You should end up with a directory containing three files:

* `FSD50K.train_mp3.hdf`
* `FSD50K.val_mp3.hdf`
* `FSD50K.eval_mp3.hdf`

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/fsd50k.py).

To fine-tune a pre-trained MobileNet on FSD50K, run the following command:

```
python ex_fsd50k.py --cuda --train --pretrained --model_name=mn10_as
```

Checkout the results of an example run [here](https://api.wandb.ai/links/florians/uyulr8l3).

To fine-tune a pre-trained DyMN on FSD50K, run the following command:

```
python ex_fsd50k.py --cuda --train --pretrained --model_name=dymn10_as --lr=0.00004 --batch_size=32
```

Checkout the results of an example run [here](https://api.wandb.ai/links/florians/3n3oxijl).


## Fine-tuning for Acoustic Scene Classification

Download the dataset *TAU Urban Acoustic Scenes 2020 Mobile, Development dataset* [11] from this [link](https://zenodo.org/record/3819968#.Y4jWjxso9GE).
Extract all files, such that you have a directory with the following content:
* *audio/* (contains all .wav files)
* *meta.csv* (contains filenames and meta data)
* *evaluation_setup/* specifies data split

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/dcase20.py).

To fine-tune a pre-trained MobileNet for acoustic scene classification, run the following command:

```
python ex_dcase20.py --cuda --pretrained --model_name=mn10_as --cache_path=cache
```

Specifying a cache path is recommended to store the resampled waveforms and avoid a bottleneck.

Checkout the results of the example run above [here](https://api.wandb.ai/links/florians/7o6g19le). 

To fine-tune a pre-trained DyMN for acoustic scene classification, run the following command:

```
python ex_dcase20.py --cuda --pretrained --model_name=dymn10_as --cache_path=cache --batch_size=32 --lr=0.0003
```

Checkout the results of the example run above [here](https://api.wandb.ai/links/florians/1e6nz5os).

## Fine-tune on ESC-50 [13]

Follow the instructions in the [PaSST](https://github.com/kkoutini/PaSST/tree/main/esc50) repository to get the ESC50 dataset.

You should end up with a folder `esc50` containing the two folders:

* `meta`: contains `meta.csv`
* `audio_32k`: contains all .wav files

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/esc50.py).

To fine-tune a pre-trained MobileNet on ESC-50, run the following command:

```
python ex_esc50.py --cuda --pretrained --model_name=mn10_as --fold=1
```

ESC-50 contains 2000 files and is divided into 5 cross-validation folds with 400 files each. The parameter `fold` specifies
which fold is used for testing.

Checkout the results of an example run [here](https://api.wandb.ai/links/florians/5r0tbm3x).

To fine-tune a pre-trained DyMN on ESC-50, run the following command:

```
python ex_esc50.py --cuda --pretrained --model_name=dymn10_as --fold=1 --lr=4e-5 --batch_size=64
```

Checkout the results of an example run [here](https://api.wandb.ai/links/florians/jfm8lpuz).

## Fine-tune on OpenMic [17]

[OpenMIC-2018](https://zenodo.org/records/1432913#.W6dPeJNKjOR) is a dataset for polyphonic instruments identification. Follow the instructions in the [PaSST](https://github.com/kkoutini/PaSST/tree/main/openmic) repository to get the 
OpenMIC-2018 dataset in the correct format.

You should end up with a directory containing two files:

* `openmic_train.csv_mp3.hdf`
* `openmic_test.csv_mp3.hdf`

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/fsd50k.py).

To fine-tune a pre-trained MobileNet on OpenMic, run the following command:

```
python ex_openmic.py --cuda --train --pretrained --model_name=mn10_as
```

Checkout the results of an example run [here](https://api.wandb.ai/links/florians/a23ye448).

To fine-tune a pre-trained DyMN on OpenMic, run the following command:

```
python ex_openmic.py --cuda --train --pretrained --model_name=dymn10_as --lr=2e-5 --batch_size=32
```

Checkout the results of an example run [here](https://api.wandb.ai/links/florians/v05iua2s). 

## References

[1] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, 
“Efficient Training of Audio Transformers with Patchout,” in Interspeech, 2022.

[2] Yuan Gong, Yu-An Chung, and James Glass, “AST: Audio Spectrogram Transformer,” in Interspeech, 2021.

[3] Ke Chen, Xingjian Du, Bilei Zhu, Zejun Ma, Taylor Berg-Kirkpatrick, and Shlomo Dubnov, 
“HTS-AT: A hierarchical token-semantic audio transformer for sound classification and detection,” in ICASSP, 2022

[4] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore,
Manoj Plakal, and Marvin Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in
ICASSP, 2017.

[5] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang,  Wenwu Wang, and Mark D. Plumbley, 
“Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE ACM Trans. Audio Speech Lang.
Process., 2020.

[6] Sergey Verbitskiy, Vladimir B. Berikov, and Viacheslav Vyshegorodtsev, “Eranns: Efficient residual audio 
neural networks for audio pattern recognition,” Pattern Recognit. Lett., 2022.

[7] Yuan Gong, Yu-An Chung, and James R. Glass, “PSLA: improving audio tagging with pretraining, sampling, 
labeling, and aggregation,” IEEE ACM Trans. Audio Speech Lang. Process., 2021.

[8] Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc V. Le, Mark Sandler, Bo Chen, Weijun Wang, Liang-Chieh Chen,
Mingxing Tan, Grace Chu, Vijay Vasudevan, and Yukun Zhu, “Searching for mobilenetv3,” in ICCV, 2019.

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei, “Imagenet: A large-scale hierarchical 
image database,” in CVPR, 2009.

[10] Jie Hu, Li Shen, and Gang Sun, “Squeeze-and-excitation networks,” in CVPR, 2018.

[11] T. Heittola, A. Mesaros, and T. Virtanen, “Acoustic scene classification in DCASE 2020 Challenge: 
generalization across devices and low complexity solutions,” in Proceedings of the 
Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020), 2020.

[12] Fonseca, E., Favory, X., Pons, J., Font, F., & Serra, X. (2021). Fsd50k: an open dataset of human-labeled sound events. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30, 829-852.

[13] Piczak, K. J. (2015, October). ESC: Dataset for environmental sound classification. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 1015-1018).

[14] Lin, J., Chen, W. M., Cai, H., Gan, C., & Han, S. (2021). Memory-efficient Patch-based Inference for Tiny Deep Learning. Advances in Neural Information Processing Systems, 34, 2346-2358.

[15] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).

[16] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei, “Beats: Audio pre-training with acoustic tokenizers,”
in Proceedings of the International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 5178–5193.

[17] Humphrey, E., Durand, S., & McFee, B. (2018, September). OpenMIC-2018: An Open Data-set for Multiple Instrument Recognition. In ISMIR (pp. 438-444).
