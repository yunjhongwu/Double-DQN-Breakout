# Double Deep Q-learning for Breakout 

A simple implementation of double deep Q-network based on [Keras](https://keras.io/) for playing [Breakout](https://en.wikipedia.org/wiki/Breakout_clone) 

In brief, deep Q-learning \[1\] parametrizes action-value functions in [Q-learning](https://en.wikipedia.org/wiki/Q-learning) with deep neural networks to circumvent problems caused by large state-space. The implemented method, double Q-learning \[2\], uses the current action instead of the maximum Q score to estimate the future reward.

![Breakout](https://cloud.githubusercontent.com/assets/6327275/18494918/6fe2843a-79e7-11e6-9c70-3ecdf5af4c54.gif)

[Feature transformation in action (Youtube)](https://youtu.be/Ef28c2bGXW0)

## How to use

 * You may type `$ python3 run.py train` in prompt to train a deep Q network.
 * You may type `$ python3 run.py demo` in prompt to see a demo (in slow motion with feature visualization).
 * **Breakout.py** contains a `Breakout` class, the *Breakout* game. Set `Breakout(player=Player())` to see that a deep Q-network plays the game. You may set `display=False` to fit a model silently (and much faster).
 * **Player.py** contains a `Player` class, including a deep neural network and the training algorithm descripted in \[1\] and \[2\].
   - `learning: bool`: run the training process every `FRAMES_PER_TRAINING` in the game.
   - `load_model`: load a model as well as pretrained weights in `serialized` folder.
   - This class is independent of *Breakout*. It can be adapted to any game with one agent and small number of discrete actions.
 * **serialized** folder contains a pretained sample model.
 
`Breakout` and `Player` classes have some static parameters. The names of the parameters are self-describing, and you may change these parameters to change to settings of the game and to tune parameters of the training algorithm. Note that the provided pretrained model may not work well in different game settings.


## Settings
#### Data preprocessing 
Each frame of the game is converted to a gray-scale image and stored as a 2-D *Numpy* array of unsigned 8-bit intgers. 

#### States
Frames are sampled every 0.009 seconds. A state of the game is deinfed as consecutive `Player.BANDWIDTH` resized frames, which is represented by a tensor of size *(Player.BANDWIDTH, frame_height, frame_width)*, which is defaulted to be *(5, 80, 80)*. 

#### Rewards
 * Hitting some bricks: +1
 * Failing to catch the ball: -1

#### Network architecture

 * Convolutional layer, 8\*8\*32, stride=(4, 4), rectifier
 * Convolutional layer, 4\*4\*64, stride=(2, 2), rectifier
 * Convolutional layer, 3\*3\*64, rectifier
 * Max-pooling layer, 2*2
 * Fully connected layer, 128, rectifier
 * Fully connected layer, 128, rectifier
 * Output layer, 3 actions
 * Squared loss `(reward - Q[action])²`


## Tuning for the pretrained model

The pretrained model was trained with 1,000,000 iterations of RMSProp (*learning rate=0.0002*, *decay factor=0.95*) and can be trainined further to improve its performance. Each iteration updated the weights with a batch of transitions of size 32, which is randomly sampled from a replay dataset of size 1,000,000 (it is very cruial to have a huge replay dataset for training deep Q networks; increase the size of the replay dataset if it is possible). The model was trained with ϵ linearly decaying from 1.0 to 0.1 over the first 500,000 and being fixed at 0.1 in the remaining iterations. The target Q network was updated every 5,000 iterations.


## Test environment 

 * Python 3.5.2
 * Keras 2.0.1
 * Matplotlib 2.0.0
 * Numpy 1.12.0
 * Pygame 1.9.3
 * Theano 0.9.0


## References

\[1\] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

\[2\] Van Hasselt, H., Guez, A., & Silver, D. (2015). Deep reinforcement learning with double Q-learning. CoRR, abs/1509.06461.
