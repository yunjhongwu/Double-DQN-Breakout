# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:53:22 2016
Implemented in Python 3.5.2
Author: Yun-Jhong Wu
"""

import numpy as np
import os
import pygame

class Breakout:
    WINDOW_SIZE = (320, 400)
    LIVES = 5
    
    BRICKS_LAYOUT = (10, 6)
    BRICKS_NUM = BRICKS_LAYOUT[0] * BRICKS_LAYOUT[1]

    BRICK_WIDTH = WINDOW_SIZE[0] // BRICKS_LAYOUT[0]
    BRICK_HEIGHT = 16
    BRICK_SCALE = BRICK_HEIGHT / BRICK_WIDTH
    BRICK_Y = 50
    
    PADDLE_WIDTH = 64
    PADDLE_HEIGHT = 8
    PADDLE_Y = 340
    PADDLE_X_MAX = WINDOW_SIZE[0] - PADDLE_WIDTH
    PADDLE_SPEED = 6

    BALL_RADIUS = 5
    BALL_X_MAX = WINDOW_SIZE[0] - BALL_RADIUS * 2
    BALL_SPEED = 3
    BALL_ACCELERATING = 2
        
    FRAMES_PER_SAMPLE = 9
    SAMPLES_BEFORE_TRAINING = 50000

    
    def __init__(self, player=None, display=False):        
        if not display:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()

        self.player = player        
        if player is None:
            print("Control the paddle by keyboard.")
            self.getAction = self.keyAction
        else:
            print("The input DQN model will control the paddle.")
            self.shot_shape = self.player.input_shape[1:]
            self.getAction = self.player.getAction

        self.screen = pygame.display.set_mode(Breakout.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.ntrials = 0
        self.is_playing = True

        self.reset(reset_all=True, log=False)
        
        playing = True
        count = 0
        
        while playing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False

            key_pressed = pygame.key.get_pressed()
            if self.move == -1: 
                self.paddle.left -= Breakout.PADDLE_SPEED
                self.paddle.left = max(self.paddle.left, 0)
            elif self.move == 1: 
                self.paddle.left += Breakout.PADDLE_SPEED
                self.paddle.left = min(self.paddle.left, 
                                       Breakout.PADDLE_X_MAX)
            
            self.nextFrame()
            pygame.display.flip()
            pygame.display.set_caption("Lives {0}, Score {1}, Trial {2}, Q = {3:.4f}".format(self.lives, self.score, self.ntrials, self.player.avgQ))

            if count % Breakout.FRAMES_PER_SAMPLE == 0:                
                self.move = self.getAction(key_pressed) - 1
                count = 0
                
                if self.player is not None:
                    self.feedToPlayer()
                
            if display:
                self.clock.tick(400)
            count += 1


    def keyAction(self, key_pressed):
        if key_pressed[pygame.K_LEFT]:
            return 0
            
        elif key_pressed[pygame.K_RIGHT]:
            return 2
            
        else:
            return 1            
    

    def reset(self, reset_all=True, log=True):
        self.move = 0
        self.dx = np.random.randint(-Breakout.BALL_SPEED, Breakout.BALL_SPEED)
        if self.dx >= 0:
            self.dx += 1
            
        self.dy = -Breakout.BALL_SPEED
        if reset_all:
            if log:
                self.log()

            self.score = 0        
            self.ntrials += 1
            self.lives = Breakout.LIVES
            self.bricks = {}

            for i in range(Breakout.BRICKS_LAYOUT[1]):
                for j in range(Breakout.BRICKS_LAYOUT[0]):
                    self.bricks[(i, j)] = pygame.Rect(
                                Breakout.BRICK_WIDTH * j + 1, 
                                Breakout.BRICK_Y + Breakout.BRICK_HEIGHT * i + 1, 
                                Breakout.BRICK_WIDTH - 2, 
                                Breakout.BRICK_HEIGHT - 2)
        
        self.paddle = pygame.Rect(Breakout.PADDLE_X_MAX // 2, 
                                  Breakout.PADDLE_Y,
                                  Breakout.PADDLE_WIDTH, 
                                  Breakout.PADDLE_HEIGHT)

        self.ball = pygame.Rect(Breakout.WINDOW_SIZE[0] // 2 - Breakout.BALL_RADIUS, 
                                Breakout.PADDLE_Y - Breakout.BALL_RADIUS * 2, 
                                Breakout.BALL_RADIUS * 2, 
                                Breakout.BALL_RADIUS * 2)

        
    def nextFrame(self):
        self.ball.left += self.dx
        self.ball.top += self.dy        
        self.is_playing = self.ball.top < Breakout.PADDLE_Y

        self.collision()

        self.screen.fill((0, 0, 0))

        for brickID in self.bricks:
            pygame.draw.rect(self.screen, (255, 255, 0), self.bricks[brickID])
            
        pygame.draw.rect(self.screen, (255, 165, 0), self.paddle)
        pygame.draw.circle(self.screen, (255, 255, 255), 
                           (self.ball.left + Breakout.BALL_RADIUS,
                            self.ball.top + Breakout.BALL_RADIUS), 
                           Breakout.BALL_RADIUS)


    def collision(self):
        if self.ball.top > Breakout.WINDOW_SIZE[1]: #Bottom (failed)
            self.lives -= 1
            self.reset(reset_all=(self.lives == 0))            
            
        elif self.ball.top < 0: #Top
            self.dy = -self.dy
            self.ball.top = 0 
            
        elif self.ball.left > Breakout.BALL_X_MAX: #Right
            self.dx = -self.dx
            self.ball.left = Breakout.BALL_X_MAX

        elif self.ball.left < 0: #Left
            self.dx = -self.dx
            self.ball.left = 0
       
        elif self.is_playing:
            if self.ball.colliderect(self.paddle):
                self.dy = -abs(self.dy)
                self.dx += Breakout.BALL_ACCELERATING * self.move
                self.ball.top = Breakout.PADDLE_Y - Breakout.BALL_RADIUS * 2
            
            else:
                brickID = self.brickCollision()
                if brickID in self.bricks:
                    del self.bricks[brickID]
                    self.score += 1


    def brickCollision(self):
        for brickID in self.bricks:
            brick = self.bricks[brickID]
            if self.ball.colliderect(brick):
                x = self.ball.left + Breakout.BALL_RADIUS - brick.left - Breakout.BRICK_WIDTH // 2
                y = self.ball.top + Breakout.BALL_RADIUS - brick.top - Breakout.BRICK_HEIGHT // 2
                
                if abs(y) < abs(x * Breakout.BRICK_SCALE):
                    self.dx = -self.dx
                else:
                    self.dy = -self.dy
                return brickID

        return None

    
    def getSnapshot(self): #(screen, action, score, is_playing)        
        shot = pygame.surfarray.array3d(pygame.transform.scale(self.screen, self.shot_shape))        
        shot = (shot[:, :, 0] * 0.298 + shot[:, :, 1] * 0.587 + shot[:, :, 2] * 0.114).T
        return (np.array(shot, dtype=np.uint8), 
                self.move + 1, self.score, 
                self.is_playing)


    def feedToPlayer(self, memorize: bool=True):
        if memorize:
            self.player.remember(self.getSnapshot())
            
        if self.player.learning:
            if self.player.memory_size > Breakout.SAMPLES_BEFORE_TRAINING:
                self.player.train(log=True)

            elif self.player.memory_size % 1000 == 0:
                self.clock.tick(500)
                print("#frames before training: {0}/{1}".format(self.player.memory_size, Breakout.SAMPLES_BEFORE_TRAINING))
         
        
    def log(self):
        with open("log/breakout_log.txt", '+a') as log:
            log.write("{0} {1}\n".format(self.ntrials, self.score))
