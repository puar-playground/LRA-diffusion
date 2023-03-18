# LRA-diffusion
This is the source code of the Label-Retrieval-Augmented Diffusion Models for learning with noisy labels.

<!-- ![CIFAR-10_TSNE](https://user-images.githubusercontent.com/123635107/214941573-02dfafbc-6e18-400d-87e6-fa604aab2501.png) -->

## 1. preparing python environment
create a virtual environment.<br />
Install and create a virtual environment for python3
```
sudo pip3 install virtualenv
python3 -m venv venv3
```
Activate the virtual environment and install requirements.<br />
```
source ./venv3/bin/activate
pip install -r requirements.txt
```

## 2. Pre-trained model
The pre-trianed SimCLR encoder for CIFAR-10 and CIFAR-100 is available at: [SimCLR models](https://drive.google.com/drive/folders/1SXzlQoOAksw349J2jnBSh5aCprDWdTQb?usp=sharing) <br />
Please download the SimCLR models and put them in to the model folder.<br />

CLIP models are available in the python package. Do not need to download manually.

## 3. Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in our experiments are provided in folder `noisy_label`.<br />
The label noise is generated by [PLC](https://github.com/AnonymousLRA/PLC/tree/master/cifar)

## 4. Run demo script for training LRA-diffusion
### CIFAR-10 and CIFAR-100<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 1000 --warmup_epochs 20
```
### Food101N and Food101<br />
The dataset should be downloaded according to the instruction here: [Food101N_data](https://github.com/puar-playground/LRA-diffusion/tree/main/Food101N_data)<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_Food101N.py --device cuda:0 --nepoch 1000 --warmup_epochs 1 --feature_dim 2048
```
### Clothing1M<br />
The dataset should be downloaded according to the instruction here: [Clothing1M_data](https://github.com/puar-playground/LRA-diffusion/tree/main/Clothing1M_data)<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_Clothing1M.py --device cuda:0 --nepoch 1000 --warmup_epochs 1 --feature_dim 4096
```

## Reference
(PLC) Progressive Label Correction:
```
@article{zhang2021learning,
  title={Learning with feature-dependent label noise: A progressive approach},
  author={Zhang, Yikai and Zheng, Songzhu and Wu, Pengxiang and Goswami, Mayank and Chen, Chao},
  journal={arXiv preprint arXiv:2103.07756},
  year={2021}
}
```


| Method | Standart | LRA-Diffusion |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Pre-trained encoder | - | SimCLR | CLIP ViT-B/32 | CLIP ViT-B/16 | CLIP ViT-L/14 |
| 10k images time | 3.96 | 9.52 | 9.13 | 17.12 | 49.98 |
| 50k images time | 20.77 | 41.31 | 39.82 | 92.34 | 303.17 |
