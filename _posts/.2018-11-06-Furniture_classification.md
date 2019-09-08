---
layout: post
title: Kaggle Competition, Furniture classification
categories: [Course]
---

For this project we participated in a kaggle competition, where the goal is to classify a furniture dataset. 

# Introduction

As part of a university course we took part in the kaggle competition [iMaterialist Challenge (Furniture) at FGVC5
](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018). For this competition we had to build a classifier in tensorflow that could label different kind of furniture. Via the course we had access to a cloud server with GPU.

In the end we made a classifier with a error of 0.24 (late submission). 

# Approach

As a kind of preprocessing we use a module from tensorflow hub. With this module we could extract a large feature vector for each image. This we saved in TFRecord file to quickly access the features in later classification. 

After extracting of the features

In the model architecture we had a module from tensorflow hub. Using this model we could extract a high dimensional feature vector for which we could build our own classifier. 




