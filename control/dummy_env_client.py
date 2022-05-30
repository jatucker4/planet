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

import gym
import json
import numpy as np
import pickle
import zlib
import zmq

from collections import OrderedDict

context = zmq.Context()
#  Socket to talk to server
print("Connecting to dummy env server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

class DummyEnvClient(object):

  def __init__(self):
    print("CREATING DUMMY ENV CLIENT - SHOULD ONLY BE HERE ONCE")
    self._random = np.random.RandomState(seed=0)
    self._step = None

  @property
  def observation_space(self):
    low = np.zeros([64, 64, 3], dtype=np.float32)
    high = np.ones([64, 64, 3], dtype=np.float32)
    spaces = {'image': gym.spaces.Box(low, high)}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    low = -np.ones([5], dtype=np.float32)
    high = np.ones([5], dtype=np.float32)
    return gym.spaces.Box(low, high)

  def reset(self):
    self._step = 0
    # obs = self.observation_space.sample()
    obs = self.get_observation()
    return obs

  def step(self, action):
    # obs = self.observation_space.sample()
    obs = self.get_observation()
    reward = self._random.uniform(0, 1)
    self._step += 1
    #done = self._step >= 1000
    done = self._step >= 50
    info = {}
    return obs, reward, done, info
  
  def get_observation(self):
    print("Sending request")
    socket.send(b"Hello")

    #  Get the reply.
    flags=0
    copy=True
    track=False
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    a = np.frombuffer(buf, dtype=md['dtype'])
    obs = a.reshape(md['shape'])
    #print(obs)

    obs_dict = OrderedDict()
    obs_dict['image'] = obs

    return obs_dict
