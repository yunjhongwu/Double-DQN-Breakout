# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:12:25 2016
Implemented in Python 3.5.2
Author: Yun-Jhong Wu
"""
import matplotlib
#matplotlib.use("Qt4Agg")

import keras.backend as K
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import os
import pygame

from matplotlib import animation, cm
from keras.layers import Dense, Convolution2D

from Breakout import Breakout


class Demo(Breakout):
    def __init__(self, player, output_frames: int=None):  
        
        os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()
        self.player = player

        self.screen = pygame.display.set_mode(Breakout.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.ntrials = 0
        self.output_frames = output_frames
        self.shot_shape = self.player.input_shape[1:]
        self.getAction = self.player.getAction

        self.reset(reset_all=True, log=False)

        self.fig = plt.figure(figsize=(20, 7), facecolor='#303030')

        self.game = plt.subplot2grid((7, 20), (0, 0), colspan=4, rowspan=5)
        self.game.set_xticks([])
        self.game.set_yticks([])
        self.field = self.game.imshow(self.getScreen())

        self.panel = plt.subplot2grid((7, 20), (5, 0), colspan=4, rowspan=2)
        self.panel.set_axis_off()
        self.text = self.panel.text(0.1, 0.3, "", size=11, color='white')

        layer_plots = [plt.subplot2grid((7, 20), (0, 5), rowspan=7, colspan=7),
                       plt.subplot2grid((7, 20), (0, 12), rowspan=4, colspan=4),
                       plt.subplot2grid((7, 20), (0, 16), rowspan=4, colspan=4),
                       plt.subplot2grid((14, 20), (9, 12), rowspan=1, colspan=8),
                       plt.subplot2grid((14, 20), (11, 12), rowspan=1, colspan=8),
                       plt.subplot2grid((14, 20), (13, 12), rowspan=1, colspan=8)]

        self.mat = []
        title = ["Conv 1", "Conv 2", "Conv 3", "Dense 1", "Dense 2", "Output"]
        cmap = cm.Greys_r
        cmap.set_bad('maroon', 1)
 
        for i, plot in enumerate(layer_plots):
            plot.set_axis_off()
            aspect = 1 if i < 3  else 0.04
            self.mat.append(plot.imshow([[0]], cmap=cmap, aspect=aspect,
                                         vmin=0, vmax=1,
                                         norm=col.PowerNorm(gamma=0.5)))
            plot.set_title(title[i], color='w')

        self.intermediate_output = K.function([self.player.Qfunc.layers[0].input], 
                                              [layer.output
                                               for layer in self.player.Qfunc.layers
                                               if isinstance(layer, Dense) or isinstance(layer, Convolution2D)])

        plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)

        self.ani = animation.FuncAnimation(self.fig, self.gamePlay, 
                                           init_func=self.gameInit, 
                                           repeat=False, blit=True,
                                           interval=1, frames=output_frames) 

        if self.output_frames:
            FFMpegWriter = animation.writers['ffmpeg']
            writer = FFMpegWriter(fps=100)
            self.ani.save('demo.mp4', writer=writer,
                          savefig_kwargs={'facecolor': '#303030'})

        plt.show()


    def gameInit(self):
        return (self.field, self.text, *self.mat)


    def gamePlay(self, t: int):
        if not self.output_frames:
            self.clock.tick(100)
        else:
            print("Frame", t)

        self.screen.fill((0, 0, 0))      
                
        if self.move == -1: 
            self.paddle.left -= Breakout.PADDLE_SPEED
            self.paddle.left = max(self.paddle.left, 0)

        elif self.move == 1: 
            self.paddle.left += Breakout.PADDLE_SPEED
            self.paddle.left = min(self.paddle.left, 
                                   Breakout.PADDLE_X_MAX)

        self.nextFrame()
        pygame.display.flip()

        if t % Breakout.FRAMES_PER_SAMPLE == 0:
            self.feedToPlayer()
            self.move = self.player.getAction() - 1

        self.field.set_array(self.getScreen())

        self.text.set_text("Frame {0}\nTrial {1}\nLives {2}\nScore {3}\nAvg. Q {4:.3f}".format(
                           t,  
                           self.ntrials,
                           self.lives,
                           self.score,
                           self.player.avgQ))

        if self.player.memory_size > self.player.BANDWIDTH * 2:
            features = self.getFeature()
            for i in range(6):
                self.mat[i].set_data(features[i])

        return (self.field, self.text, *self.mat)


    def getScreen(self): #(screen, action, score, is_playing)        
        return np.rollaxis(pygame.surfarray.array3d(self.screen), 1, 0)


    def flatten(self, z, nrows: int, ncols: int):
        z = z[0, :]
        size = z.shape[1] + 1
        feature = np.empty((size * nrows - 1, size * ncols - 1)) * 0.3
        feature[:] = np.NAN
        for k in range(z.shape[0]):
            i = k // ncols
            j = k % ncols
            feature[i * size:(i + 1) * size - 1, 
                    j * size:(j + 1) * size - 1] = z[k, :, :]
        return feature


    def getFeature(self):
        z = self.intermediate_output([self.player.getCurrent()])
        for i in range(5):
            z[i] -= np.min(z[i])
            z[i] *= 1 / np.max(z[i])

        z[0] = self.flatten(z[0], 4, 8)
        z[1] = self.flatten(z[1], 8, 8)
        z[2] = self.flatten(z[2], 8, 8)
        z[5] = np.exp(10 * z[5])
        z[5] /= np.sum(z[5])
        return z
