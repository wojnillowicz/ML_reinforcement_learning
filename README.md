# Machine Learning: reinforcement learning
Machine Learning project

------------------------------------------------------------
Falling Object with Reinforcement Learning (Q-Learning)
------------------------------------------------------------
This project implements a simple reinforcement learning agent
that learns to catch a falling object using tabular Q-learning.

The environment is a 1D vertical field where:
- An object ('*') falls straight down in a fixed column.
- A basket ('^') at the bottom can move left, stay, or move right.

The goal is for the basket to catch the object when it reaches
the bottom row. If it does, the agent receives a reward of +1,
otherwise -1. During training, rewards are only given at the end.

The agent observes the current state, which consists of:
- The x-position of the falling object.
- The x-position of the basket.

The agent uses a Q-table to learn optimal actions over time.
Training happens over many episodes, and the success rate is
tracked and visualized using matplotlib at the end.

After training, the agent plays a few rounds where you can
watch the object fall in the terminal and see if it is caught.


------------------------------------------------------------
Requirements
------------------------------------------------------------

Make sure you have the following Python libraries installed:

- matplotlib
- numpy
