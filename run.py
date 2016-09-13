# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:32:13 2016
Implemented in Python 3.5.2
Author: Yun-Jhong Wu
"""

import sys


from Breakout import Breakout
from Player import Player
from Demo import Demo


def train_dqn(display=False):
    """
    Fit the model in a reinforcement learning manner
    """
    player = Player(learning=True, load_model=True)
    return Breakout(player=player, display=display)


def demo(output_frames=None):
    """
    See the result of a trained model
    """
    player = Player(learning=False, load_model=True)
    return Demo(player=player, output_frames=output_frames)


if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Training a model...")
        train_dqn(display=(len(sys.argv) > 2 and sys.argv[2] == "display"))

    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("Demo in slow motion with feature visualization")
        demo(output_frames=None)
       
    else:
        print("Unknown option")
