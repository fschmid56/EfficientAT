# Efficient Large-Scale Audio Tagging

In this repository, we publish pre-trained models and code described in the paper [Efficient Large-Scale Audio Tagging 
Via Transformer-To-CNN Knowledge Distillation](https://arxiv.org/pdf/2211.04772.pdf). The paper is submitted to 
[ICASSP 2023](https://2023.ieeeicassp.org/). 

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

**This codebase is under construction** and will change in the following weeks. The milestones are:
* Provide Audio Tagging models pre-trained on AudioSet **[Done]**
* Show how the pre-trained models can be loaded for inference **[Done]**
* Add pre-trained models that work on different spectrogram resolutions
* Include Training Models on AudioSet from scratch
* Include Fine-Tuning of AudioSet pre-trained models on downstream tasks
* Evaluate the quality of extracted embeddings and figure out at which layer to obtain the most powerful embeddings
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



**Pre-trained models are contained in the Github Releases and must be placed in the 'resources' folder.** 
Loading the pre-trained models is as easy as running the following code piece:

```
from models.MobileNetV3 import get_model
model = get_model(width_mult=1.0, pretrained_name="mn10_as")
```

The Table shows all models contained in this repository. The naming convention for our models is 
**mn_\<width_mult\>_\<dataset\>**. In this sense, *mn10_as* defines a MobileNetV3 with parameter *width_mult=1.0*, pre-trained on 
AudioSet.

All models available are pre-trained on ImageNet [9], followed by training on AudioSet [4]. The results appear slightly better than those reported in the
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
| mn10_as_hop_15   | width_mult=1.0                                     | 4.88              | 0.36            | .463              |
| mn10_as_hop_20   | width_mult=1.0                                     | 4.88              | 0.27            | .456              |
| mn10_as_hop_25   | width_mult=1.0                                     | 4.88              | 0.22            | .447              |
| mn10_as_mels_40  | width_mult=1.0                                     | 4.88              | 0.21            | .453              |
| mn10_as_mels_64  | width_mult=1.0                                     | 4.88              | 0.27            | .461              |
| mn10_as_mels_256 | width_mult=1.0                                     | 4.88              | 1.08            | .474              |

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

**Important:** All models are trained with half precision (float16). If you run float32 inference on cpu,
you might notice a slight performance degradation.

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




