# LRA-diffusion
This is the source code of the Label-Retrieval-Augmented Diffusion Models for Noise Label Learning.

<!-- ![CIFAR-10_TSNE](https://user-images.githubusercontent.com/123635107/214941573-02dfafbc-6e18-400d-87e6-fa604aab2501.png) -->

## 1. preparing python environment
Install requirements.<br />
```
pip install -r requirements.txt
```

## 2. Pre-trained model
The pre-trianed SimCLR encoder for CIFAR-10 and CIFAR-100 is available at: [SimCLR models](https://github.com/puar-playground/LRA-diffusion/tree/main/model) <br />
Please download the SimCLR models and put them in to the model folder.<br />

CLIP models are available in the python package.

## 3. Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in our experiments are provided in folder `noisy_label`.<br />
The label noise is generated by [PLC](https://github.com/puar-playground/PLC/tree/master/cifar)

## 4. Run demo script for training LRA-diffusion
### CIFAR-10 and CIFAR-100<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_CIFAR.py --device cuda:0 --noise_type cifar10-1-0.35 --fp_encoder SimCLR --nepoch 200 --warmup_epochs 5
```
### Food101N and Food101<br />
The dataset should be downloaded according to the instruction here: [Food101N](https://github.com/puar-playground/LRA-diffusion/tree/main/Food101N_data)<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_Food101N.py --device cuda:0 --nepoch 32 --warmup_epochs 1 --feature_dim 1024
```
### Clothing1M<br />
The dataset should be downloaded according to the instruction here: [Clothing1M](https://github.com/puar-playground/LRA-diffusion/tree/main/Clothing1M_data)<br />
Default values for input arguments are given in the code. The [training data](https://github.com/puar-playground/LRA-diffusion/tree/main/Clothing1M/annotations) is selected by the pre-trained ["Centrality and Consistency"](https://github.com/uitrbn/tscsi_idn) classification model. An example command is given:
```
python train_Clothing1M.py --gpu_devices 0 1 2 3 --nepoch 200 --warmup_epochs 1 --feature_dim 1024
```

## Reference


