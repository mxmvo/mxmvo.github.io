---
layout: post
title: Saliency in the Arcade
categories: [Course]
thumbnail: "/assets/images/saliency_arcade/pong.jpg"
---

Where does and agent 'look', does it base its decision on the same information as humans? 

The investigation of the 'saliency' of an reinforcement learning agent is based on the article of [Greydanus](http://arxiv.org/abs/1711.00138). They introduce a method that describes the amount of information in a certain area of the playing field has. This results in a measure of 'saliency'. In this experiment we let people play pong. A human saliency map is constructed from the extracted eye-tracking data. Then we will see the relation between the agents' saliency and the human's saliency. For the agent we could already use a pretrained model from [Greydanus](http://arxiv.org/abs/1711.00138).

![Eye Tracking](/assets/images/saliency_arcade/human.gif){:height="300px", width="200px"} ![Machine Saliency](/assets/images/saliency_arcade/machine_saliency.gif){:height="300px" width="200px"}

# Results
Using the KL-divergence as a method of dissimilarty between saliency maps. We see that the human's are in general looking at the same area as the agent is extracting its information from. The agents only is not always 'looking' when it does not have any affect on the game, while the humans are in general following the ball. Furthermore an agent can 'look' at multiple areas aswell. These does results do effect the eventual results. 

# Side Quest
Could we use the eye-tracking data from the humans as a catalyst for the agents learning method. Inspiration came from [article](http://arxiv.org/abs/1806.03960), so initially we made a saliency predictor and precomposed this with an agent. 
## Saliency Predictor
For the saliency prediction we used the human-saliency a a ground truth and we tried to predict where the human was looking. As architecture the image of the playing field was convoluted down three times, the get an abstract representation, from the representation we used the transpose convolution to get the original sized image again. Instead of a memory cell, we gave the predictor 4 sequential images. As a cost function we used the KL-Divergence with regularization, as optimizer the Adam optimizer. 

![Saliency Predictor](/assets/images/saliency_arcade/saliency_pred.png)

## Agent with saliency

For the agent to use the saliency data we precompose the training with the saliency prediction. Afterwards the predicted saliency is implemented in the prediction.

![Saliency Predictor](/assets/images/saliency_arcade/agent_sal_architecture.png)

Unfortunately the prediction of the saliency in serie with the training greatly increaded the computation time, this didn't allow us to train and fine tune this model in the time allowed. 

## Side note
We expect that the Pong game is to simplistic to benefit from saliency data. Since there is not that much going on. We are hopeful, but further research is needed, that in more complex games where more planning is needed and a lot of information is presented on the screen saliency data might help boost the training. 