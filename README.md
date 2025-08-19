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
conda create -n physiosync python=3.9
conda activate physiosync
pip install -r requirements.txt
```

## 📊 Dataset
We evaluate PhysioSync on benchmark EEG-based emotion recognition datasets.
Please download the datasets from their official sources
- [DEAP](https://eecs.qmul.ac.uk/mmv/datasets/deap/)
- [DREAMER](https://zenodo.org/records/546113)


## 🏃Pre-training and Fine-tuning
Take DEAP's "Dependent" as an example

### Training
```bash
python main_pretrain_1s.py
python main_pretrain_5s.py
```
### Fine-tuning

```bash
cd Fine_tuning
python fine_tuning.py 
```


## Citation
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

