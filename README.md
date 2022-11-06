# A viable framework for semi‑supervised learning on realistic dataset (MindSpore)


## The Implementation with MindSpore

**Paper Link:**   https://link.springer.com/article/10.1007/s10994-022-06208-6

## Introduction
Semi-supervised Fine-Grained Recognition is a challenging task due to the difculty of 
data imbalance, high inter-class similarity and domain mismatch. Recently, this feld has 
witnessed giant leap and many methods have gained great performance. We discover that 
these existing Semi-supervised Learning (SSL) methods achieve satisfactory performance 
owe to the exploration of unlabeled data. However, on the realistic large-scale datasets, 
due to the abovementioned challenges, the improvement of the quality of pseudo-labels 
requires further research. In this work, we propose Bilateral-Branch Self-Training Framework (BiSTF), a simple yet efective framework to improve existing semi-supervised learning methods on class-imbalanced and domain-shifted fne-grained data. By adjusting stochastic epoch update frequency, BiSTF iteratively retrains a baseline SSL model with a 
labeled set expanded by selectively adding pseudo-labeled samples from an unlabeled set, 
where the distribution of pseudo-labeled samples is the same as the labeled data. We show 
that BiSTF outperforms the existing state-of-the-art SSL algorithm on Semi-iNat dataset. 
	
### Experiments Results of CIFAR10-Semi

| Method            | beta = 10%  | beta = 30%       | beta = 80% |  
|:--------------------|:------------|:------------    |:------------  |
| Pseudo-labeling            | 68.15    | 75.94   | 77.98              |
| Mean teacher     | 76.28    | 80.89  | 81.58              |
| MixMatch    | 78.91    | 82.46  | 83.85             |
| FixMatch (RA)         | 81.15    | 84.51  | 86.26           |
| BiSTF   | 80.02       | 85.16  | 88.47             |

## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (for distributed mode) 

To install the dependency, please run
```shell
pip install -r requirements.txt
```

## Train and Validation

### Train

``` shell
python train.py --model=resnet50 --dataset=your_data_path --val_while_train --val_split=val --val_interval=1 --ckpt_save_dir your_save_path
```

You can add more parameters in the configs file by reading the paper, such as RandAugment etc.

### Validation

```python
python validate.py --model=resnet50 --dataset=your_data_path --val_split=validation --ckpt_path='./ckpt/resnet50-best.ckpt' 
``` 

### Acknowledgement

This work is sponsored by Natural Science Foundation of China(62276242), CAAI-Huawei MindSpore Open
Fund(CAAIXSJLJJ-2021-016B), Anhui Province Key Research and Development Program(202104a05020007), and
USTC Research Funds of the Double First-Class Initiative(YD2350002001)

### Citation

If you find this project useful in your research, please consider citing:

```latex
@article{chang2022viable,
  title={A viable framework for semi-supervised learning on realistic dataset},
  author={Chang, Hao and Xie, Guochen and Yu, Jun and Ling, Qiang and Gao, Fang and Yu, Ye},
  journal={Machine Learning},
  pages={1--23},
  year={2022},
  publisher={Springer}
}
```
