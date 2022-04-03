# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import numpy as np
import os
import pickle
import random
import gym
from collections import OrderedDict
from planet.control.abstract import AbstractEnvironment
from planet.humanav_examples.examples import *


def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)

class Stanford_Environment_Params():
    def __init__(self):
        self.epi_reward = 100 #1000 #100
        self.step_reward = -1 #-10 #-1
        self.dim_action = 1
        self.velocity = 0.2 #0.1 #0.5 
        self.dim_state = 2
        self.img_size = 32 
        self.dim_obs = 4
        self.obs_std_light = 0.01
        self.obs_std_dark = 0.1
        self.step_range = 1 #0.05
        self.max_steps = 200  
        self.noise_amount = 0.4 #1.0 #0.4 #0.15
        self.occlusion_amount = 15
        self.salt_vs_pepper = 0.5
        self.fig_format = '.png'
        self.img = 'data/img/'
        self.ckpt = 'data/ckpt/'

        #self.training_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_hallway/rgbs/*"
        #self.testing_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_hallway/rgbs/*"
        # self.training_data_path = "/home/sampada/training_hallway/rgbs/*"
        # self.testing_data_path = "/home/sampada/testing_hallway/rgbs/*"
        self.training_data_path = "/home/sampada/vae_stanford/training_hallway/rgbs/*"
        self.testing_data_path = "/home/sampada/vae_stanford/testing_hallway/rgbs/*"
        self.normalization = True

sep = Stanford_Environment_Params()


class IntermediateDummyEnv(object):

    def __init__(self, disc_thetas=False):
        #self._random = np.random.RandomState(seed=0)
        self._step = None

        #self.done = False
        self.reached_goal = False
        self.true_env_corner = [24.0, 23.0] 
        self.xrange = [0, 8.5]
        self.yrange = [0, 1.5]
        self.thetas = [0.0, 2*np.pi]
        self.disc_thetas = disc_thetas
        self.discrete_thetas = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        self.discrete_thetas = self.discrete_thetas.reshape((len(self.discrete_thetas), 1))
        self.trap_x = [[1.5, 2], [6.5, 7]] 
        self.trap_y = [0, 0.25]
        self.target_x = [4, 4.5] 
        self.target_y = [0, 0.25]

        # During test time - have an additional trap region (optional)
        self.test_trap = False
        self.test_trap_is_random = False
        self.test_trap_x = [[3, 3.5], [5, 5.5]] #[[1.5, 2], [6.5, 7]] #[3.5, 5]
        self.test_trap_y = [[0.5, 1], [0.5, 1]] #[0.5, 1] #[0.75, 1.25]

        self.init_strip_x = self.xrange 
        self.init_strip_y = [0.25, 0.5]
        self.state, self.orientation = self.initial_state()
        #self.state, self.orientation = np.random.rand(sep.dim_state), np.random.rand() 
        self.dark_line = (self.yrange[0] + self.yrange[1])/2
        self.dark_line_true = self.dark_line + self.true_env_corner[1]

        # Get the traversible
        try:
            traversible = pickle.load(open("traversible.p", "rb"))
            dx_m = 0.05
        except Exception:
            path = os.getcwd() + '/temp/'
            os.mkdir(path)
            _, _, traversible, dx_m = self.get_observation(path=path)
            pickle.dump(traversible, open("traversible.p", "wb"))

        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]

        # For making training batches
        self.training_data_path = sep.training_data_path
        self.training_data_files = glob.glob(self.training_data_path)
        self.testing_data_path = sep.testing_data_path
        self.testing_data_files = glob.glob(self.testing_data_path)
        self.normalization = sep.normalization
    

    def initial_state(self):
        if self.disc_thetas:
            orientation = self.discrete_thetas[np.random.randint(len(self.discrete_thetas))][0]
        else:
            orientation = np.random.rand()  
            orientation = orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]
 
        valid_state = False
        while not valid_state:
            state = np.random.rand(sep.dim_state)
            temp = state[1]
            state[0] = state[0] * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
            state[1] = temp * (self.trap_y[1] - self.trap_y[0]) + self.trap_y[0]  # Only consider x for in_trap
            if not self.in_trap(state) and not self.in_goal(state):
                valid_state = True
                state[1] = temp * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]

        return state, orientation 
    
    def in_trap(self, state):
        # Returns true if in trap
        first_trap = self.trap_x[0]
        first_trap_x = (state[0] >= first_trap[0] and state[0] <= first_trap[1])
        second_trap = self.trap_x[1]
        second_trap_x = (state[0] >= second_trap[0] and state[0] <= second_trap[1])
        trap_x = first_trap_x or second_trap_x

        trap = trap_x and (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) 

        # Traps that may appear during test time -- optional
        if self.test_trap:
            first_test_trap = self.test_trap_x[0]
            first_test_trap_x = (state[0] >= first_test_trap[0] and state[0] <= first_test_trap[1])
            second_test_trap = self.test_trap_x[1]
            second_test_trap_x = (state[0] >= second_test_trap[0] and state[0] <= second_test_trap[1])
            #test_trap_x = first_test_trap_x or second_test_trap_x

            first_test_trap = self.test_trap_y[0]
            first_test_trap_y = (state[1] >= first_test_trap[0] and state[1] <= first_test_trap[1])
            second_test_trap = self.test_trap_y[1]
            second_test_trap_y = (state[1] >= second_test_trap[0] and state[1] <= second_test_trap[1])
            
            test_trap = (first_test_trap_x and first_test_trap_y) or \
                        (second_test_trap_x and second_test_trap_y)

            # test_trap = test_trap_x and (state[1] >= self.test_trap_y[0] and state[1] <= self.test_trap_y[1])
            
            return (trap or test_trap) and not self.in_goal(state)
        
        return trap and not self.in_goal(state)
    
    def in_goal(self, state):
        # Returns true if in goal
        goal = (state[0] >= self.target_x[0] and state[0] <= self.target_x[1]) and \
                (state[1] >= self.target_y[0] and state[1] <= self.target_y[1])
        return goal


    @property
    def observation_space(self):
        low = np.zeros([64, 64, 3], dtype=np.float32)
        high = np.ones([64, 64, 3], dtype=np.float32)
        spaces = {'image': gym.spaces.Box(low, high)}
        return gym.spaces.Dict(spaces)

    # @property
    # def action_space(self):
    #     low = -np.ones([5], dtype=np.float32)
    #     high = np.ones([5], dtype=np.float32)
    #     return gym.spaces.Box(low, high)
    
    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    # def reset(self):
    #     self._step = 0
    #     obs = self.observation_space.sample()
    #     return obs
    
    def reset(self):
        #self.done = False
        self.reached_goal = False
        self._step = 0
        self.state, self.orientation = self.initial_state()

        # Randomizing the test traps: 
        # Each trap is size 0.5 by 0.5
        # Trap 1 will be randomly placed between x = 0 and 4 (goal x)
        # Trap 2 will be randomly placed between x = 4.5 and 8 (end of goal x to end of xrange - 0.5)
        # Each trap will be randomly placed between y = 0.5 and 0.75 (dark line)

        if self.test_trap and self.test_trap_is_random:
            trap_size = 0.5
            trap1_x = np.random.rand() * (self.target_x[0] - self.xrange[0]) + self.xrange[0]
            trap2_x = np.random.rand() * (self.xrange[1]-trap_size - self.target_x[1]) + self.target_x[1]
            self.test_trap_x = [[trap1_x, trap1_x+trap_size], [trap2_x, trap2_x+trap_size]] 

            trap1_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            trap2_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            self.test_trap_y = [[trap1_y, trap1_y+trap_size], [trap2_y, trap2_y+trap_size]]
        
        obs = self.observation_space.sample()

        return obs

    # def step(self, action):
    #     obs = self.observation_space.sample()
    #     reward = np.random.uniform(0, 1)
    #     #reward = self._random.uniform(0, 1)
    #     self._step += 1
    #     #done = self._step >= 1000
    #     done = self._step >= 50
    #     info = {}
    #     return obs, reward, done, info
    
    def step(self, action, action_is_vector=False):
        episode_length = 50

        #self.done = False
        curr_state = self.state

        obs = self.observation_space.sample()
        self._step += 1
        
        if action_is_vector:
            new_theta = np.arctan2(action[1], action[0])
            if new_theta < 0:  # Arctan stuff
                new_theta += 2*np.pi
            next_state = curr_state + action
        else:
            new_theta = action[0] * np.pi + np.pi
            vector = np.array([np.cos(new_theta), np.sin(new_theta)]) * sep.velocity  # Go in the direction the new theta is
            next_state = curr_state + vector
    
        cond_hit = self.detect_collision(next_state)

        # Previous value of reached_goal 
        temp_reached_goal = self.reached_goal

        # If already reached goal, don't move
        if not temp_reached_goal:
            if self.in_goal(next_state):
                self.state = next_state
                self.orientation = new_theta
                self.reached_goal = True
                #self.done = True
            elif cond_hit == False:  # If collided, don't move. Else move.
                self.state = next_state
                self.orientation = new_theta
        #reward = sep.epi_reward * self.done

        # If we just reached the goal, collect reward
        # But if we already reached the goal previously, get reward 0
        if not temp_reached_goal and self.reached_goal:
            reward = sep.epi_reward 
        else:
            reward = 0
        
        # If already reached goal, don't reason about trap rewards
        if not temp_reached_goal:
            cond_false = self.in_trap(next_state)
            reward -= sep.epi_reward * cond_false

        done = self._step >= episode_length

        info = {}
        return obs, reward, done, info
    
    def point_to_map(self, pos_2, cast_to_int=True):
        """
        Convert pos_2 in real world coordinates
        to a point on the map.
        """
        map_pos_2 = pos_2 / self.dx - self.map_origin
        if cast_to_int:
            map_pos_2 = map_pos_2.astype(np.int32)
        return map_pos_2

    def detect_collision(self, state):
        # Returns true if you collided
        # Map value 0 means there's an obstacle there

        # Don't hit rectangle boundaries
        if state[0] < self.xrange[0] or state[0] > self.xrange[1]:
            return True 
        if state[1] < self.yrange[0] or state[1] > self.yrange[1]:
            return True

        # Check if state y-value is the same as trap/goal but it's not in the trap or goal - that's a wall
        # if self.in_trap([self.trap_x[0][0], state[1]]) and \
        #     not self.in_trap(state) and not self.in_goal(state):
        #     return True
        if (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) and \
            not self.in_trap(state) and not self.in_goal(state):
            return True

        map_state = self.point_to_map(np.array(state[:sep.dim_state] + self.true_env_corner))
        map_value = self.traversible[map_state[1], map_state[0]]
        collided = (map_value == 0)
        return collided


'''
class IntermediateDummyEnv(object):

    def __init__(self, disc_thetas=False):
        #self._random = np.random.RandomState(seed=0)

        self.done = False
        self._step = None
        self.true_env_corner = [24.0, 23.0] 
        self.xrange = [0, 8.5]
        self.yrange = [0, 1.5]
        self.thetas = [0.0, 2*np.pi]
        self.disc_thetas = disc_thetas
        self.discrete_thetas = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        self.discrete_thetas = self.discrete_thetas.reshape((len(self.discrete_thetas), 1))
        self.trap_x = [[1.5, 2], [6.5, 7]] 
        self.trap_y = [0, 0.25]
        self.target_x = [4, 4.5] 
        self.target_y = [0, 0.25]

        # During test time - have an additional trap region (optional)
        self.test_trap = False
        self.test_trap_is_random = False
        self.test_trap_x = [[3, 3.5], [5, 5.5]] #[[1.5, 2], [6.5, 7]] #[3.5, 5]
        self.test_trap_y = [[0.5, 1], [0.5, 1]] #[0.5, 1] #[0.75, 1.25]

        self.init_strip_x = self.xrange 
        self.init_strip_y = [0.25, 0.5]
        self.state, self.orientation = self.initial_state()
        self.dark_line = (self.yrange[0] + self.yrange[1])/2
        self.dark_line_true = self.dark_line + self.true_env_corner[1]

        # Get the traversible
        try:
            traversible = pickle.load(open("traversible.p", "rb"))
            dx_m = 0.05
        except Exception:
            path = os.getcwd() + '/temp/'
            os.mkdir(path)
            # _, _, traversible, dx_m = self.get_observation(path=path)
            pickle.dump(traversible, open("traversible.p", "wb"))

        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]

        # # For making training batches
        # self.training_data_path = sep.training_data_path
        # self.training_data_files = glob.glob(self.training_data_path)
        # self.testing_data_path = sep.testing_data_path
        # self.testing_data_files = glob.glob(self.testing_data_path)
        # self.normalization = sep.normalization
  
    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    @property
    def observation_space(self):
        # low = np.ones([64, 64, 3], dtype=np.float32)*-10
        # high = np.ones([64, 64, 3], dtype=np.float32)*10
        low = np.zeros([64, 64, 3], dtype=np.float32)
        high = np.ones([64, 64, 3], dtype=np.float32)
        spaces = {'image': gym.spaces.Box(low, high)}
        return gym.spaces.Dict(spaces)

    # def reset(self):
    #     self._step = 0
    #     obs = self.observation_space.sample()
    #     return obs

    def reset(self):
        self.done = False
        self._step = 0
        self.state, self.orientation = self.initial_state()

        # Randomizing the test traps: 
        # Each trap is size 0.5 by 0.5
        # Trap 1 will be randomly placed between x = 0 and 4 (goal x)
        # Trap 2 will be randomly placed between x = 4.5 and 8 (end of goal x to end of xrange - 0.5)
        # Each trap will be randomly placed between y = 0.5 and 0.75 (dark line)

        if self.test_trap and self.test_trap_is_random:
            trap_size = 0.5
            trap1_x = np.random.rand() * (self.target_x[0] - self.xrange[0]) + self.xrange[0]
            trap2_x = np.random.rand() * (self.xrange[1]-trap_size - self.target_x[1]) + self.target_x[1]
            self.test_trap_x = [[trap1_x, trap1_x+trap_size], [trap2_x, trap2_x+trap_size]] 

            trap1_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            trap2_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            self.test_trap_y = [[trap1_y, trap1_y+trap_size], [trap2_y, trap2_y+trap_size]]

        obs = self.observation_space.sample()

        return obs

    # def step(self, action):
    #     obs = self.observation_space.sample()
    #     # reward = self._random.uniform(0, 1)
    #     reward = np.random.uniform(0, 1)
    #     self._step += 1
    #     done = self._step >= 1000
    #     info = {}
    #     return obs, reward, done, info
  
    def step(self, action, action_is_vector=False):  ## TODO Add time horizon!
        self.done = False
        curr_state = self.state

        obs = self.observation_space.sample()

        if action_is_vector:
            new_theta = np.arctan2(action[1], action[0])
            if new_theta < 0:  # Arctan stuff
                new_theta += 2*np.pi
            next_state = curr_state + action
        else:
            new_theta = action[0] * np.pi + np.pi
            vector = np.array([np.cos(new_theta), np.sin(new_theta)]) * sep.velocity  # Go in the direction the new theta is
            next_state = curr_state + vector

        cond_hit = self.detect_collision(next_state)

        if self.in_goal(next_state):
            self.state = next_state
            self.orientation = new_theta
            self.done = True
        elif cond_hit == False:
            self.state = next_state
            self.orientation = new_theta
        reward = sep.epi_reward * self.done

        cond_false = self.in_trap(next_state)
        reward -= sep.epi_reward * cond_false

        info = {}
        self._step += 1
        return obs, reward, self.done, info

    def detect_collision(self, state):
        # Returns true if you collided
        # Map value 0 means there's an obstacle there

        # Don't hit rectangle boundaries
        if state[0] < self.xrange[0] or state[0] > self.xrange[1]:
            return True 
        if state[1] < self.yrange[0] or state[1] > self.yrange[1]:
            return True

        # Check if state y-value is the same as trap/goal but it's not in the trap or goal - that's a wall
        # if self.in_trap([self.trap_x[0][0], state[1]]) and \
        #     not self.in_trap(state) and not self.in_goal(state):
        #     return True
        if (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) and \
            not self.in_trap(state) and not self.in_goal(state):
            return True

        map_state = self.point_to_map(np.array(state[:sep.dim_state] + self.true_env_corner))
        map_value = self.traversible[map_state[1], map_state[0]]
        collided = (map_value == 0)
        return collided

    def point_to_map(self, pos_2, cast_to_int=True):
        """
        Convert pos_2 in real world coordinates
        to a point on the map.
        """
        map_pos_2 = pos_2 / self.dx - self.map_origin
        if cast_to_int:
            map_pos_2 = map_pos_2.astype(np.int32)
        return map_pos_2

    def initial_state(self):
        if self.disc_thetas:
            orientation = self.discrete_thetas[np.random.randint(len(self.discrete_thetas))][0]
        else:
            orientation = np.random.rand()  
            orientation = orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]

        valid_state = False
        while not valid_state:
            state = np.random.rand(sep.dim_state)
            temp = state[1]
            state[0] = state[0] * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
            state[1] = temp * (self.trap_y[1] - self.trap_y[0]) + self.trap_y[0]  # Only consider x for in_trap
            if not self.in_trap(state) and not self.in_goal(state):
                valid_state = True
                state[1] = temp * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]

        return state, orientation 

    def in_trap(self, state):
        # Returns true if in trap
        first_trap = self.trap_x[0]
        first_trap_x = (state[0] >= first_trap[0] and state[0] <= first_trap[1])
        second_trap = self.trap_x[1]
        second_trap_x = (state[0] >= second_trap[0] and state[0] <= second_trap[1])
        trap_x = first_trap_x or second_trap_x

        trap = trap_x and (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) 

        # Traps that may appear during test time -- optional
        if self.test_trap:
            first_test_trap = self.test_trap_x[0]
            first_test_trap_x = (state[0] >= first_test_trap[0] and state[0] <= first_test_trap[1])
            second_test_trap = self.test_trap_x[1]
            second_test_trap_x = (state[0] >= second_test_trap[0] and state[0] <= second_test_trap[1])
            #test_trap_x = first_test_trap_x or second_test_trap_x

            first_test_trap = self.test_trap_y[0]
            first_test_trap_y = (state[1] >= first_test_trap[0] and state[1] <= first_test_trap[1])
            second_test_trap = self.test_trap_y[1]
            second_test_trap_y = (state[1] >= second_test_trap[0] and state[1] <= second_test_trap[1])
            
            test_trap = (first_test_trap_x and first_test_trap_y) or \
                        (second_test_trap_x and second_test_trap_y)

            # test_trap = test_trap_x and (state[1] >= self.test_trap_y[0] and state[1] <= self.test_trap_y[1])
            
            return (trap or test_trap) and not self.in_goal(state)

        return trap and not self.in_goal(state)

    def in_goal(self, state):
        # Returns true if in goal
        goal = (state[0] >= self.target_x[0] and state[0] <= self.target_x[1]) and \
                (state[1] >= self.target_y[0] and state[1] <= self.target_y[1])
        return goal
'''
