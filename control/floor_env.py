from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from planet.config.floor import *
from planet import training
import numpy as np


class FloorEnv(object):

  def __init__(self):
    self.done = False
    self.state = np.random.rand(2)
    self.target1 = np.array([2, 0.25])
    self.target2 = np.array([0, 0.75])
    self.false_target1 = np.array([0, 0.25])
    self.false_target2 = np.array([2, 0.75])
    self.walls_x = [[1.2, 0.9, 1], [0.4, 0.4, 0.5], [1.2, 0.5, 0.6], [0.4, 0.0, 0.1]]
    self.walls_y = [[0.5, 0, 2]]
    self.steps = 0

  @property
  def action_space(self):
    return gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([0.5, 0.5]), dtype=np.float32)

  def observation_space(self):
    low = np.array([0.0, 0.0, 0.0, 0.0])
    high = np.ones([2.0, 2.0, 1.0, 1.0])
    spaces = {'image': gym.spaces.Box(low=low, high=high, dtype=np.float32)}
    return gym.spaces.Dict(spaces)

  def get_observation(self):
    x = self.state[0]
    y = self.state[1]
    obs_x1 = x
    obs_y1 = y
    obs_x2 = 1 - x
    obs_y2 = 1 - y
    num_wall_x = len(self.walls_x)
    num_wall_y = len(self.walls_y)
    for i in range(num_wall_x):
      wx = self.walls_x[i][0]
      wy1 = self.walls_x[i][1]
      wy2 = self.walls_x[i][2]
      if y > wy1 and y < wy2 and x > wx:
        dist_x1 = x - wx
        obs_x1 = min(obs_x1, dist_x1)
      if y > wy1 and y < wy2 and x < wx:
        dist_x2 = wx - x
        obs_x2 = min(obs_x2, dist_x2)
    for i in range(num_wall_y):
      wy = self.walls_y[i][0]
      wx1 = self.walls_y[i][1]
      wx2 = self.walls_y[i][2]
      if x > wx1 and x < wx2 and y > wy:
        dist_y1 = y - wy
        obs_y1 = min(obs_y1, dist_y1)
      if x > wx1 and x < wx2 and y < wy:
        dist_y2 = wy - y
        obs_y2 = min(obs_y2, dist_y2)
    obs = np.array([obs_x1, obs_y1, obs_x2, obs_y2])
    obs += np.random.normal(0, 0.01, DIM_OBS)
    return obs

  def step(self, action):
    self.done = False
    self.steps += 1
    curr_state = self.state
    next_state = curr_state + action

    cond = (curr_state[1] <= 0.5)
    target = cond * self.target1 + (1 - cond) * self.target2

    next_dist = training.utility.l2_distance(next_state, target)
    cond_hit = training.utility.detect_collison(curr_state, next_state)

    if next_dist <= END_RANGE or self.steps >= 1000:
      self.state = next_state
      self.done = True
    elif cond_hit == False:
      self.state = next_state
    reward = EPI_REWARD * self.done

    false_target = cond * self.false_target1 + (1 - cond) * self.false_target2
    curr_false_dist = training.utility.l2_distance(curr_state, false_target)
    next_false_dist = training.utility.l2_distance(next_state, false_target)
    cond_false = (curr_false_dist >= END_RANGE) * (next_false_dist < END_RANGE)
    reward -= EPI_REWARD * cond_false
    obs = self.get_observation()
    info = {}
    return obs, reward, self.done, info

  def reset(self):
    self.state[0] = self.state[0] * 0.4 + 0.8
    self.state[1] = self.state[1] * 0.3 + 0.1 + np.random.randint(2) * 0.5
    obs = self.get_observation()
    return obs
