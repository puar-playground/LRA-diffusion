# Label-Retrieval-Augmented Diffusion Models for Learning from Noisy Labels
source code of the [LRA-Diffusion](https://arxiv.org/abs/2305.19518v2) Models for Learning from Noisy Labels.

<!-- ![CIFAR-10_TSNE](https://user-images.githubusercontent.com/123635107/214941573-02dfafbc-6e18-400d-87e6-fa604aab2501.png) -->

## 1. Preparing python environment
Install requirements.<br />
```
pip install -r requirements.txt
```

## 2. Pre-trained model & Checkpoints
* The pre-trianed SimCLR encoder for CIFAR-10 and CIFAR-100 is provided in the [model](https://github.com/puar-playground/LRA-diffusion/tree/main/model) folder. <br />
* CLIP models are available in the python package at [here](https://github.com/openai/CLIP). Install without dependency: <br />
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git  --no-dependencies
```
* For Clothing1M, the pre-trained ["Centrality and Consistency"](https://github.com/uitrbn/tscsi_idn) (CC) classification model is also provided.

Trained checkpoints for the diffusion models are available at [here](https://drive.google.com/drive/folders/1SXzlQoOAksw349J2jnBSh5aCprDWdTQb?usp=share_link).

## 3. Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in our experiments are provided in folder `noisy_label`. The noisy labels are generated following the original [paper](https://openreview.net/pdf?id=ZPa2SyGcbwh).

## 4. Run demo script to train the LRA-diffusion
### 4.1 CIFAR-10 and CIFAR-100<br />
Default values for input arguments are provided in the code. An example command is:
```
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 200 --warmup_epochs 5
```
### 4.2 Food-101N and Food-101<br />
The dataset should be downloaded according to the instruction here: [Food-101N](https://github.com/puar-playground/LRA-diffusion/tree/main/Food101N)<br />
Default values for input arguments are provided in the code. An example command is:
```
python train_Food101N.py --gpu_devices 0 1 2 3 --nepoch 200 --warmup_epochs 1 --feature_dim 1024
```
### 4.3 Clothing1M<br />
The dataset should be downloaded according to the instruction here: [Clothing1M](https://github.com/puar-playground/LRA-diffusion/tree/main/Clothing1M_data). Default values for input arguments are provided in the code. <br />

The [training data](https://github.com/puar-playground/LRA-diffusion/tree/main/Clothing1M/annotations) is selected by the pre-trained CC classifier. An example command using multiple gpus is:
```
python train_Clothing1M.py --gpu_devices 0 1 2 3 --nepoch 200 --warmup_epochs 1 --feature_dim 1024
```
### 4.4 WebVision<br />
Download [WebVision 1.0](https://data.vision.ee.ethz.ch/cvl/webvision/download.html) and the validation set of [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) datasets. The ImageNet synsets labels for ILSVRC2012 validation set is provided [here](https://github.com/puar-playground/LRA-diffusion/tree/main/ILSVRC2012).
```
python train_WebVision.py --gpu_devices 0 1 2 3 --nepoch 200 --warmup_epochs 1 --feature_dim 1024
```

## Reference
```
@article{chen2024label,
  title={Label-retrieval-augmented diffusion models for learning from noisy labels},
  author={Chen, Jian and Zhang, Ruiyi and Yu, Tong and Sharma, Rohan and Xu, Zhiqiang and Sun, Tong and Chen, Changyou},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

