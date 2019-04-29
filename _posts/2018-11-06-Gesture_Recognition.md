---
layout: post
title: Gesture Recognition
categories: [Course]
thumbnail: "/assets/images/gesture_recog/thumbnail.png"
---

This project was done in  collabaration with @vhrgitai and @faaip, for the repo see [github](https://github.com/faaip/OpenPose-Gesture-Recognition).

# OpenPose Gesture Recognition

In this project we made a classifier, to classify the videos from the [20bn jester dataset](https://20bn.com/datasets/jester)
This dataset contains videos with gestures made by people from all over the world, by using their webcam. 


To preprocess the videos (seen as a series of images), we used [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/) posture recognition from Carnegie Mellon University's Perceptual Computing Lab. This way each video was converted to a sequence of body-point registrations. 

A video below shows an example of the extracted body points projected back on the original video, as you can see not every image get's succesfully parsed.

![Example video](/assets/images/gesture_recog/animation.gif) ![parsed video](/assets/images/gesture_recog/processed.gif)


# Training the Classifier

The architecture of the RNN is based on [guillaume-chevalier](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition). The architecture consists of an fully connected layer followed by two LSTM cell, then the eventual classification is made by two fully connected layers again. 

The cost function is the cross entropy function for the prediction together with a regularization term to keep the model from overfitting, this function was optimized using the AdamOptimizer.

For training of the classifier we took our dataset and randomly seperate it in a training set of $14000$ and a test set of $3054$.


# Results

The classifier ended up with an accuracy off roughly $50\%$. We want to emphasize that we used only $10\%$ of the total dataset, this was because of computational limitations. The extaction of the datapoints from the videos took longer that expected. Our hope is that the accuracy will increase with the size of the dataset. Training the classifier on more data will hopefully make it able to distringuish more between similar gestures. The confusion matrix shows what kind of mistakes the classifier was making. 

![Confussion Matrix](/assets/images/gesture_recog/confusion_matrix_all.png){:heigth="700px" width="700px"}
