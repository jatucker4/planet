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

from planet.control import wrappers

from planet import control

def random_episodes(env_ctor, num_episodes, outdir=None):
  import tensorflow as tf
  import numpy as np
  
  env = env_ctor()
  env = wrappers.CollectGymDataset(env, outdir)
  #env = wrappers.CollectGymDataset.get_my_env(env, outdir)
  # env = wrappers.ConcatObservation(env, ['image'])
  #env = control.batch_env.BatchEnv([env], blocking=False)
  # env = control.in_graph_batch_env.InGraphBatchEnv(env)
  episodes = [] if outdir else None
  for _ in range(num_episodes):
    policy = lambda env, obs: env.action_space.sample()
    done = False
    obs = env.reset()
    while not done:
      action = policy(env, obs)
      # print("ACTION", action, type(action))
      obs, _, done, info = env.step(action)
      # obs, _, done, info = env.step([action])
      # obs, _, done, info = env.step(np.array([action]))
      # obs, _, done, info = env.step(tf.reshape(tf.convert_to_tensor([action]), [1, 1]))
    if outdir is None:
      episodes.append(info['episode'])
  try:
    env.close()
  except AttributeError:
    pass
  # print("FINISHED RANDOM EPISODE COLLECTION")
  return episodes
