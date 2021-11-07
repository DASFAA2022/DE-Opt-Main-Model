# DE-Opt-Main-Model
This implemention is for the paper work of "Toward Auto-learning Hyperparameters for deep learning-based Recommender Systems"
## Brief Introduction
This paper proposes a general hyperparameter optimization framework for existing DL-based RSs based on differential evolution (DE), named as DE-Opt. The main idea of DE-Opt is to incorporate DE into a DL-based recommendation model’s training process to auto-learn its hyperparameters λ (regulariza-tion coefficient) and η (learning rate) simultaneously at layer-granularity. Thereby, its performance of both recommendation accuracy and computa-tional efficiency is boosted. Empirical studies on three benchmark datasets verify that: 1) DE-Opt can significantly improve state-of-the-art DL-based recommendation models by making their λ and η adaptive, and 2) DE-Opt also significantly outperforms state-of-the-art DL-based recommendation models whose λ and/or η are/is adaptive. 
## Enviroment Requirement
The Code has been tested running under Python 3.7.0
requires:
tensorflow
numpy
scipy
sklearn
pandas
## Dataset
This code contains Movilens 1M dataset for the example running
