---
layout: post
title: Reinforcement Learning with Lego Mindstorm
categories: [Course]
thumbnail: /assets/images/lego_crawler/thumbnail.png
---

Reinforment Learning combined with Lego Mindstorm. 

As a free project I had the opportunity to apply reinforcement methods using Lego Mindstorm. This project has been very informative, it allowed me to brush up on different Reinforcement Learning methods. But also it showed the gap between software and hardware. Normally I have been able to simulate games on the computer, but now the simulation has to be done in real life. This severely slowed down training as one might expect. 

My personal goal was to build a crawler, the building blocks was contained in a education Mindstorm box. 

# Design

The design of the crawler contain a brick with two motors. The two motors make the arm move, here I used gears to make the arm able to bent (like an elbow). The red L shaped bits are there to initialize the motors to their starting position. On front there is a infra-red meter, this is used to measure distance and calculate the reward function. By connecting to the intelligent Brick via bluetooth I was able to control the robot from my computer. This way all the calculation/training could be done remotely and not on the brick, which would significantly increase training time.

![jpg](/assets/images/lego_crawler/lego_crawler_1.JPG){:height="360px" width="720apx"}

![jpg](/assets/images/lego_crawler/lego_crawler_2.JPG){:height="360px" width="720apx"}


# Model

A working agent was made using tabular Q-learning, the action and state were made discrete. This made learning relatively quick. For the discretization the agent could move each motor with a certain angle clockwise or counter clockwise, it also had the option of doing nothing. So there where 8 different actions (both doing nothing didn't add anything). The state space was discretized into bins one set of bins for angle of a motor, the width of the bins was 25 degrees. New states where dynamically added to the Q-table, resulting in 24 different states. This might be less than expected but it is because the motors do not have a free range of movement. See the below video for a working example of the model

![Example video](/assets/images/lego_crawler/lego_robot.gif)

The rubber band and the mirror were added to gain more friction. As the gif shows the movements of the robot, are discrete. 


## Extra

Furthermore I experimented with a continuous state space and a discrete action space. As a training algorithm I used the Agent Critic method. Using this method unfortunately didn't succeed. A possible (and I think likely) reason is that the simulations were too slow, potentially training the model longer could help. Additionally the motors were not precise enough. Meaning that if the motor had a angle of 100 degrees and you want to increase it by 20, then it ended up being somewhere between 125 and 115. For the discrete state space this is not that bad since it would still be binned the same. For the continuous state space this is extra noise that it has to deal with. 

