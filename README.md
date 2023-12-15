# TextGCN

This project was done as part of Associated Shared Task in the FOURTH WORKSHOP ON LANGUAGE TECHNOLOGY FOR EQUALITY, DIVERSITY, INCLUSION (LT-EDI-2024) AT EACL 2024

## Task

Caste/Immigration Hate Speech Detection in Tamil Text

## Introduction

This repository contains the code for the paper [TextGCN: A Text Graph Convolutional Network for Text Classification](https://arxiv.org/abs/1809.05679) (AAAI 2019).

We used that model in featureless to train on the tamil dataset and detect hate speech.

As performance was not good, we used SBert to provide sentence embeddings and then used TextGCN to train on the embeddings.

## Folder Structure

```
.
├── data
│   ├── train.csv  : Training data
│   ├── dev.csv    : Validation data
│   ├── test.csv   : Test data

├── model
│   ├── with embedding : Model with SBert embeddings
│   │   ├── model.pt
│   ├── without embedding : Model without SBert embeddings
│   │   ├── model.pt

├── utils
│   ├── length.txt : Size of the dataset
│   ├── TamilStopWords.txt : Stopwords for Tamil
│   ├── vocab.csv : Vocabulary of the dataset


├── embedding : Contains the embedding obtained before the final layer of the model
│   ├── Train_embedding.csv
│   ├── Dev_embedding.csv
│   ├── Test_embedding.csv

├── results
│   ├── result.txt : Results of the models
│   ├── pred.csv : Predictions of the model on the test data
│   ├── Metric Graphs : Graphs of various metrics during training
│   ├── tSNE Graphs : tSNE graphs of the embeddings

├── code
│   ├── textgcn.py : Code for the model
│   ├── graph_construction.ipynb : Code for constructing the graph
│   ├── train.ipynb : Code for training the model

```

## Model Architecture and Hyperparameters

First layer turn 768 dimensional SBert embeddings to 64 dimensional embeddings.

Second layer turns 64 dimensional embeddings to 16 dimensional embeddings.

Third layer turns 16 dimensional embeddings to 2 dimensional embeddings.

Optimizer : Adam

Loss : Cross Entropy

Epoch : 36

Hyperparameter:learning rate 0.01 and dropout 0.5


## Results

| Model | Accuracy | F1 Score |
| --- | --- | --- |
| TextGCN | 0.69 | 0.68 |
| TextGCN with SBert Embeddings | 0.78 | 0.76 |











