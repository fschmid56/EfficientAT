# Efficient Large-Scale Audio Tagging

In this repository, we publish pre-trained models and code described in the paper [Efficient Large-Scale Audio Tagging 
Via Transformer-To-CNN Knowledge Distillation](https://arxiv.org/pdf/2211.04772.pdf). The paper is submitted to 
[ICASSP 2023](https://2023.ieeeicassp.org/). 

The models in this repository are especially suited to you if you are looking for audio classification models that are able to:
* achieve **competitive performance on resource constrained platforms**
* extract **high-quality general purpose audio representations**
* reach **high performance on downstream tasks with scarce audio data**

Large-scale Audio Tagging is dominated by Transformers (PaSST [1], AST [2], HTS-AT [3]) achieving the highest 
single-model mean average precisions (mAP) on AudioSet [4]. However, Transformers are complex
models and scale quadratically with respect to the sequence length, making them slow for inference.
CNNs scale linearly with respect to the sequence length and are easy to scale to given resource constraints. 
However, CNNs (e.g. PANNs [5], ERANN [6], PSLA [7]) have fallen short on Transformers in terms of Audio Tagging performance.

**We bring together the best of both worlds by training efficient CNNs of different complexity using Knowledge Distillation 
from Transformers**. The Figures below show the performance-complexity trade-off for existing Audio Tagging models in
comparison to our proposed models based on the MobileNetV3 [8] architecture.

![Model Performance vs. Model Size](/images/model_params.png)

![Model Performance vs. Computational Complexity](/images/model_macs.png)

The next milestones are:
* Provide Audio Tagging models pre-trained on AudioSet **[Done]**
* Show how the pre-trained models can be loaded for inference **[Done]**
* Add pre-trained models that work on different spectrogram resolutions **[Done]**
* Add an easy way to ensemble models **[Done]**
* Include Training and Evaluating Models on AudioSet **[Done]**
* Include Fine-Tuning of AudioSet pre-trained models on downstream tasks **[Done]**
* Evaluate the quality of extracted embeddings and figure out at which layer to obtain the most powerful embeddings **[Done]**
* Provide pre-training routine on ImageNet

The final repository should have similar capabilities as the [PANNs codebase](https://github.com/qiuqiangkong/audioset_tagging_cnn)
with two main advantages:
* Pre-trained models of **lower computational and parameter complexity** due to the efficient MobileNetV3 CNN architecture
* **Higher performance** due to Knowledge Distillation from Transformers

This codebase is inspired by the [PaSST](https://github.com/kkoutini/PaSST) and 
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) repositories, and the [pytorch implementation of
MobileNetV3](https://pytorch.org/vision/stable/models/mobilenetv3.html).

## Environment

The codebase is developed with *Python 3.10.8*. After creating an environment install the requirements as
follows:

```
pip install -r requirements.txt
```

## Pre-Trained Models



**Pre-trained models are available in the Github Releases and are automatically downloaded from there.** 
Loading the pre-trained models is as easy as running the following code piece:

```
from models.MobileNetV3 import get_model
model = get_model(width_mult=1.0, pretrained_name="mn10_as")
```

The Table shows all models contained in this repository. The naming convention for our models is 
**mn_\<width_mult\>_\<dataset\>**. In this sense, *mn10_as* defines a MobileNetV3 with parameter *width_mult=1.0*, pre-trained on 
AudioSet.

All models available are pre-trained on ImageNet [9] by default (otherwise denoted as 'no_im_pre'), followed by training on AudioSet [4]. The results appear slightly better than those reported in the
paper. We provide the best models in this repository while the paper is showing averages over multiple runs.

| Model Name       | Config                                             | Params (Millions) | MACs (Billions) | Performance (mAP) |
|------------------|----------------------------------------------------|-------------------|-----------------|-------------------|
| mn04_as          | width_mult=0.4                                     | 0.983             | 0.11            | .432              |
| mn05_as          | width_mult=0.5                                     | 1.43              | 0.16            | .443              |
| mn10_as          | width_mult=1.0                                     | 4.88              | 0.54            | .471              |
| mn20_as          | width_mult=2.0                                     | 17.91             | 2.06            | .478              |
| mn30_as          | width_mult=3.0                                     | 39.09             | 4.55            | .482              |
| mn40_as          | width_mult=4.0                                     | 68.43             | 8.03            | .484              |
| mn40_as_ext      | width_mult=4.0,<br/>extended training (300 epochs) | 68.43             | 8.03            | .487              |
| mn40_as_no_im_pre      | width_mult=4.0, no ImageNet pre-training           | 68.43             | 8.03            | .483              |
| mn10_as_hop_15   | width_mult=1.0                                     | 4.88              | 0.36            | .463              |
| mn10_as_hop_20   | width_mult=1.0                                     | 4.88              | 0.27            | .456              |
| mn10_as_hop_25   | width_mult=1.0                                     | 4.88              | 0.22            | .447              |
| mn10_as_mels_40  | width_mult=1.0                                     | 4.88              | 0.21            | .453              |
| mn10_as_mels_64  | width_mult=1.0                                     | 4.88              | 0.27            | .461              |
| mn10_as_mels_256 | width_mult=1.0                                     | 4.88              | 1.08            | .474              |
| Ensemble         | width_mult=4.0, 9 Models                           | 615.87            | 72.27           | .498              |

Ensemble denotes an ensemble of 9 different mn40 models (3x mn40_as,
3x mn40_as_ext, 3x mn40_as_no_im_pre). 

The Parameter and Computational complexity (number of multiply-accumulates) is calculated using the script [complexity.py](complexity.py). Note that the number of MACs calculated with our procedure is qualitatively as it counts only the dominant operations (linear layers, convolutional layers and attention layers for Transformers). 

The complexity statistics of a model can be obtained by running:

```
python complexity.py --model_name="mn10_as"
```

Which will result in the following output:

```
Model 'mn10_as' has 4.88 million parameters and inference of a single 10-seconds audio clip requires 0.54 billion multiply-accumulate operations.
```

Note that computational complexity strongly depends on the resolution of the spectrograms. Our default is 128 mel bands and a hop size of 10 ms.

## Inference

You can use one of the pre-trained models for inference on a an audio file using the 
[inference.py](inference.py) script.  

For example, use **mn10_as** to detect acoustic events at a metro station in paris:

```
python inference.py --cuda --model_name=mn10_as --audio_path="resources/metro_station-paris.wav"
```

This will result in the following output showing the 10 events detected with the highest probability:

```
************* Acoustic Event Detected: *****************
Train: 0.811
Rail transport: 0.649
Railroad car, train wagon: 0.630
Subway, metro, underground: 0.552
Vehicle: 0.328
Clickety-clack: 0.179
Speech: 0.061
Outside, urban or manmade: 0.029
Music: 0.026
Train wheels squealing: 0.021
********************************************************
```

You can also use an ensemble for perform inference, e.g.:

```
python inference.py --ensemble mn40_as_ext mn40_as mn40_as_no_im_pre --cuda --audio_path=resources/metro_station-paris.wav
```


**Important:** All models are trained with half precision (float16). If you run float32 inference on cpu,
you might notice a slight performance degradation.

## Quality of Extracted Embeddings 

The full analysis and code for the HEAR benchmark evaluation is located in the [EfficientAT_HEAR](https://github.com/fschmid56/EfficientAT_HEAR) repository.
Below you find a short presentation of the results.

To evalute the quality of extracted audio embeddings as general purpose audio representations, we run
the [HEAR Benchmark](https://hearbenchmark.com/) on our models. The benchmark consists of 19 different tasks. In the first
step embeddings are computed for all 19 tasks using a single model. In the second step, a shallow down-stream MLP is trained
on the embeddings to solve the tasks. The performance of the shallow MLP on the downstream tasks is an indicator for the quality of extracted embeddings.

The following plot compares the scores of PANNs (CNN14) [5], PaSST [1] and our models on each task. The score is normalized by
the maximal performance reached by a model on the official [leaderboard](https://hearbenchmark.com/hear-leaderboard.html).

![Hear Task Performances](/images/hear_individual_task_results.png)

The plot shows that **mn40_as_ext** achieves a new highscore on the tasks *Gunshot Triangulation*,
*ESC-50* and *GTZAN Genre* across all models on the [leaderboard](https://hearbenchmark.com/hear-leaderboard.html). The quality of embeddings is in general comparable
to PaSST embeddings. **mn40_as_ext** outperforms **PANNs (CNN14)** on *11/19* tasks, and PANNs has a higher score on
*5/19* tasks (no results are reported for PANNs on the remaining 3 tasks).

The tiny model **mn04_as** (1 Mil. parameters) performs exceptionally well compared to PANNs (CNN14, 80 Mil. params) 
achieving higher scores on *6/19* and lower scores on *9/19* tasks (one tie).

Checkout the results summarized per task category for each model:

![Hear Task Category Performances](/images/hear_task_categories.png)

## Train and Evaluate on AudioSet

The training and evaluation procedures are simplified as much as possible. The most difficult part is to get AudioSet[4]
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
python ex_audioset.py --cuda --model_name="mn10_as"
```

Which will result in the following output:

```
Results on AudioSet test split for loaded model: mn10_as
  mAP: 0.471
  ROC: 0.980
```

#### Training

Logging is done using [Weights & Biases](https://wandb.ai/site). Create a free account to log your experiments. During training
the latest model will be saved to the directory [wandb](wandb).

To train a model on AudioSet, you can run for example the following command:
```
python ex_audioset.py --cuda --train --pretrained_name=mn10_im_pytorch --batch_size=60 --max_lr=0.0004
```

Checkout the results of this example configuration [here](https://wandb.ai/florians/EfficientAudioTagging/reports/Training-mn10_as-from-ImageNet-pre-trained-on-a-GeForce-RTX-2080-Ti--VmlldzozMDMwMTc4).

To train a tiny model (```model_width=0.1```) with Squeeze-and-Excitation [10] on the frequency dimension and a fully convolutional
classification head, run the following command:

```
python ex_audioset.py --cuda --train --batch_size=120 --model_width=0.1 --head_type=fully_convolutional --se_dims=f
```

Checkout the results of this example configuration [here](https://wandb.ai/florians/EfficientAudioTagging/reports/Train-Tiny-Model-width-0-1---VmlldzozMDMwMjkx).

## Fine-tuning for Acoustic Scene Classification

Download the dataset *TAU Urban Acoustic Scenes 2020 Mobile, Development dataset* [11] from this [link](https://zenodo.org/record/3819968#.Y4jWjxso9GE).
Extract all files, such that you have a directory with the following content:
* *audio/* (contains all .wav files)
* *meta.csv* (contains filenames and meta data)
* *evaluation_setup/* specifies data split

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/dcase20.py).

To fine-tune a pre-trained model for acoustic scene classification run the following command:

```
python ex_dcase20.py --cuda --pretrained_name=mn10_as --cache_path=<directory>
```

Specifying a cache path is recommended to store the resampled waveforms and avoid a bottleneck. 

The lightweight ```mn10_as``` can be fine-tuned on a GeForce RTX 2080 Ti
in less than 15 minutes to around 70% accuracy, which is 90% of PaSST [1] SOTA performance (76,3% accuracy).
``mn40_as`` achieves an accuracy of 74% with the default fine-tuning procedure. Tuning the hyperparameters in
[ex_dcase20.py](ex_dcase20.py) might lead to higher accuracies.

Checkout the results of the example run above [here](https://wandb.ai/florians/DCASE20/reports/Fine-Tuning-mn10_as-for-acoustic-scene-classification--VmlldzozMDcyNDk0?accessToken=6mflvehphxqybp64aw3n2fa6b603tw6voq60oy8bys3b5pb0fc9qgvxnxw1gqhx5).

## Fine-tune on FSD50K [12]

Follow the instructions in the [PaSST](https://github.com/kkoutini/PaSST/tree/main/fsd50k) repository to get the FSD50K dataset.

You should end up with three files:

* `FSD50K.train_mp3.hdf`
* `FSD50K.val_mp3.hdf`
* `FSD50K.eval_mp3.hdf`

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/fsd50k.py).

To fine-tune a pre-trained model on FSD50K run the following command:

```
python ex_fsd50k.py --cuda --pretrained_name=mn10_as
```

Checkout the results of an example run [here](https://wandb.ai/florians/FSD50K/reports/Fine-tune-mn10_as-on-FSD50K--VmlldzozMTk0MzY5?accessToken=4mx9x8k2mgq424r4gjkcu0c2ipmk8irbdtz7bafvk97w1a71y5z67f7nj76l03gs).

## Fine-tune on ESC-50 [13]

Follow the instructions in the [PaSST](https://github.com/kkoutini/PaSST/tree/main/esc50) repository to get the ESC50 dataset.

You should end up with a folder `esc50` containing the two folders:

* `meta`: contains `meta.csv`
* `audio_32k`: contains all .wav files

Specify the location of this directory in the variable ```dataset_dir``` in the [dataset file](datasets/esc50.py).

To fine-tune a pre-trained model on ESC-50 run the following command:

```
python ex_esc50.py --cuda --pretrained_name=mn40_as_ext --fold=1
```

ESC-50 contains 2000 files and is divided into 5 cross-validation folds with 400 files each. The parameter `fold` specifies
which fold is used for testing.

Checkout the results of an example run [here](https://wandb.ai/florians/ESC50/reports/-mn40_as_ext-on-ESC50--VmlldzozMTk1NDYz?accessToken=f7s0b7h7ve6jtp8mc42exhj94pio8228udcl4ug6z4o1irrgtfeocz3b7c32exgh).

While our models lag behind [PaSST](https://github.com/kkoutini/PaSST/) transformer performance on FSD50K due its large scale
(around 50k audio samples) our models outperform PaSST on ESC-50 since the fine-tuning data is scarce (only 2000 samples).

Compared to the results presented in the official [ESC-50 github repo](https://github.com/karolpiczak/ESC-50), we achieve a new SOTA accuracy of 97,45% with
`mn40_as_ext` as pre-trained model.

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