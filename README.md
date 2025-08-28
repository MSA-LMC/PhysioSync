# PhysioSync: Temporal and Cross-Modal Contrastive Learning Inspired by Physiological Synchronization for EEG-based Emotion Recognition

[![arXiv](https://img.shields.io/badge/arXiv-2504.17163-b31b1b.svg)](https://arxiv.org/abs/2504.17163)


## 🚀 Introduction
PhysioSync is a pre-training framework for EEG-based emotion recognition that leverages **temporal and cross-modal contrastive learning** inspired by physiological synchronization.  
It incorporates **Cross-Modal Consistency Alignment (CM-CA)** to capture dynamic relationships between EEG and Peripheral Physiological Signals (PPS), and **Long- and Short-Term Temporal Contrastive Learning (LS-TCL)** to model emotional fluctuations across different time resolutions.  
Experiments on **DEAP** and **DREAMER** demonstrate that PhysioSync achieves robust improvements in both uni-modal and cross-modal settings.

## 📦Create environment

We recommend Python >= 3.8.

Install dependencies:

```bash
pip install -r requirements.txt
```
(Alternatively, if you use conda:)

```bash
conda create -n physiosync python=3.8
conda activate physiosync
pip install -r requirements.txt
```

## 📊 Dataset
We evaluate PhysioSync on benchmark EEG-based emotion recognition datasets.
Please download the datasets from their official sources
- [DEAP](https://eecs.qmul.ac.uk/mmv/datasets/deap/)
- [DREAMER](https://zenodo.org/records/546113)


## 🏃Pre-Training and Fine-Tuning
Take DEAP's "Dependent" as an example

### Pre-Training
```bash
python main_pretrain_1s.py
python main_pretrain_5s.py
```
### Fine-Tuning

```bash
cd Fine_tuning
python fine_tuning.py 
```

## 📊 Experimental Results

We report results under **Dependent** (within-subject) and **Cross-subject** evaluation protocols for DEAP and DREAMER datasets.

### 1️⃣ Dependent (Within-Subject) Results
![Dependent DEAP Results](figures/dependent_deap.png)
*Dependent DEAP results*

![Dependent DREAMER Results](figures/dependent_dreamer.png)
*Dependent DREAMER results*

### 2️⃣ Cross-Subject Results

![Cross-Subject DEAP Results](figures/cross_subject.png)
*Cross-subject results*


## 📖Citation
```
@misc{cui2025physiosynctemporalcrossmodalcontrastive,
      title={PhysioSync: Temporal and Cross-Modal Contrastive Learning Inspired by Physiological Synchronization for EEG-Based Emotion Recognition}, 
      author={Kai Cui and Jia Li and Yu Liu and Xuesong Zhang and Zhenzhen Hu and Meng Wang},
      year={2025},
      eprint={2504.17163},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.17163}, 
}
```

