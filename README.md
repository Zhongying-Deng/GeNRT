
# GeNRT
Pytorch implementation of GeNRT: [Zhongying Deng, Da Li, Junjun He, Xiaojiang Peng, Yi-Zhe Song, Tao Xiang. "Generative models for noise-robust training in unsupervised domain adaptation." Pattern Recognition (2026)](https://doi.org/10.1016/j.patcog.2025.112450).


## Installation

- Please first install the [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started) as follows (or refer to [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started) for more details):


Make sure [conda](https://www.anaconda.com/distribution/) is installed properly. The following installation commands are adapted from [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started).

```bash
# Clone this repo
git clone https://github.com/Zhongying-Deng/GeNRT.git
cd GeNRT

# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch and torchvision (select a version that suits your machine)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, `pytorch 1.7.1 + cuda 10.1, python 3.7` should be installed. **Note that installing Dassl is a must.**

- Follow the instructions in [DATASETS.md](./DATASETS.md) to install the datasets. All the data should be stored under the `data` folder.

- You may also be interested in my other implementations of unsupervised domain adaptation work: [DAC-Net (BMVC 2021)](https://github.com/Zhongying-Deng/DAC-Net), [DIDA-Net (IEEE T-IP 2022)](https://github.com/Zhongying-Deng/DIDA), [BORT2 (Pattern Recognition 2026)](https://github.com/Zhongying-Deng/BORT2). They are also implemented using [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started).

### Training

- The training scripts are provided in the bash files, such as `train_digit5_genrt.sh` for training on Digit-Five using GeNRT (similar names apply to PACS). The backbone models for the FixMatch-CM can be found at `dassl/modeling/backbone`, such as `resnet18_nflow_cmix.py` or `cnn_digit5_m3sda_nflow_cmix.py`, all with the suffix ‘_nflow_cmix.py’.

- The trainer for Digit5 is named  `FixMatchNFlowClassMixConsistencyDigit5`, of which the Python file can be found at `dassl/engine/da/fixmatch_nflow_class_mix_consistency_digit5.py`, while for PACS, the trainer is `FixMatchNFlowClassMixConsistency` with `dassl/engine/da/fixmatch_nflow_class_mix_consistency.py` as its corresponding Python file.

- The config files are also specified in the script. `configs/datasets/da/digit5_nflow.yaml` provides the configurations of the Digit5 dataset, while `configs/trainers/ssl/fixmatch_ema/digit5.yaml` specifies the hyperparameters for model optimization. Similar config files apply to the PACS dataset.

- The checkpoints and log files will be saved in the `output` folder.

## Citation
Please cite the following paper if you find Dassl useful to your research.

```
@article{DENG2026112450,
title = {Generative models for noise-robust training in unsupervised domain adaptation},
journal = {Pattern Recognition},
volume = {172},
pages = {112450},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112450},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325011124},
author = {Zhongying Deng and Da Li and Junjun He and Xiaojiang Peng and Yi-Zhe Song and Tao Xiang},
keywords = {Unsupervised domain adaptation, Multi-source domain adaptation, GeNRT, Normalizing flow, Feature augmentation},
}
```