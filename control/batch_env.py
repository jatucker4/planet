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

import numpy as np
import pickle
import time

#from examples.examples import *  # generate_observation
from planet.humanav_examples.examples import *


class BatchEnv(object):
  """Combine multiple environments to step them in batch."""

  batchenv = None

  def __init__(self, envs, blocking):
    """Combine multiple environments to step them in batch.

    To step environments in parallel, environments must support a
    `blocking=False` argument to their step and reset functions that makes them
    return callables instead to receive the result at a later time.

    Args:
      envs: List of environments.
      blocking: Step environments after another rather than in parallel.

    Raises:
      ValueError: Environments have different observation or action spaces.
    """
    self._envs = envs
    #print("\nI got here, here are the envs!", self._envs, blocking, "\n")
    self._blocking = blocking
    observ_space = self._envs[0].observation_space
    if not all(env.observation_space == observ_space for env in self._envs):
      raise ValueError('All environments must use the same observation space.')
    action_space = self._envs[0].action_space
    if not all(env.action_space == action_space for env in self._envs):
      raise ValueError('All environments must use the same observation space.')


  # def __init__(self, factory, env_ctor, num_agents, blocking):
  #   """Combine multiple environments to step them in batch.

  #   To step environments in parallel, environments must support a
  #   `blocking=False` argument to their step and reset functions that makes them
  #   return callables instead to receive the result at a later time.

  #   Args:
  #     envs: List of environments.
  #     blocking: Step environments after another rather than in parallel.

  #   Raises:
  #     ValueError: Environments have different observation or action spaces.
  #   """
  #   self._envs = [factory(env_ctor) for _ in range(num_agents)]
  #   print("\nI got here, here are the envs!", self._envs, blocking, "\n")
  #   self._blocking = blocking
  #   observ_space = self._envs[0].observation_space
  #   if not all(env.observation_space == observ_space for env in self._envs):
  #     raise ValueError('All environments must use the same observation space.')
  #   action_space = self._envs[0].action_space
  #   if not all(env.action_space == action_space for env in self._envs):
  #     raise ValueError('All environments must use the same observation space.')
  
  @classmethod
  def get_my_env(cls, envs, blocking):
  #def get_my_env(cls, factory, env_ctor, num_agents, blocking):
    """
    Used to instantiate a BatchEnv object. Ensures that only one BatchEnv
    object ever exists.
    """
    b = cls.batchenv
    if b is not None:
      return b 

    cls.batchenv = cls(envs, blocking)
    #cls.batchenv = cls(factory, env_ctor, num_agents, blocking)
    return cls.batchenv

  def __len__(self):
    """Number of combined environments."""
    return len(self._envs)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._envs[index]

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name one of the wrapped environments.
    """
    return getattr(self._envs[0], name)

  def timer(self, arg):
    t = time.time()
    print("Batch env timer", t)
    return t

  def step(self, actions):
    """Forward a batch of actions to the wrapped environments.

    Args:
      actions: Batched action to apply to the environment.

    Raises:
      ValueError: Invalid actions.

    Returns:
      Batch of observations, rewards, and done flags.
    """
    for index, (env, action) in enumerate(zip(self._envs, actions)):
      if not env.action_space.contains(action):
        #print("ENV ACTION SPACE", env.action_space)
        message = 'Invalid action at index {}: {}'
        raise ValueError(message.format(index, action))
    if self._blocking:
      transitions = [
          env.step(action)
          for env, action in zip(self._envs, actions)]
    else:
      print("\nGoing to enter env.step now\n")
      t0 = time.time()
      try:
          pickle_time0 = time.time()
          planning_times = pickle.load(open("planning_times.p", "rb"))
          print("Elapsed time", t0-planning_times[-1])
          planning_times.append(t0)
          pickle.dump(planning_times, open("planning_times.p", "wb"))
          pickle_time1 = time.time()
          print("Pickling time", pickle_time1-pickle_time0)
      except Exception:
          planning_times = [t0]
          pickle.dump(planning_times, open("planning_times.p", "wb"))

      transitions = [
          env.step(action, blocking=False)
          for env, action in zip(self._envs, actions)]
      # transitions = [
      #     env.step(action)
      #     for env, action in zip(self._envs, actions)]
      transitions = [transition() for transition in transitions]
    observs, rewards, dones, infos = zip(*transitions)
    observ = np.stack(observs)
    reward = np.stack(rewards).astype(np.float32)
    done = np.stack(dones)
    info = tuple(infos)
    return observ, reward, done, info

  def reset(self, indices=None):
    """Reset the environment and convert the resulting observation.

    Args:
      indices: The batch indices of environments to reset; defaults to all.

    Returns:
      Batch of observations.
    """
    if indices is None:
      indices = np.arange(len(self._envs))
    if self._blocking:
      observs = [self._envs[index].reset() for index in indices]
    else:
      print("\nGoing to enter env.reset now\n")
      #print(self._envs)
      observs = [self._envs[index].reset(blocking=False) for index in indices]
      #observs = [self._envs[index].reset() for index in indices]
      #print("OBSERVS", observs)
      observs = [observ() for observ in observs]
    observ = np.stack(observs)
    return observ

  def close(self):
    """Send close messages to the external process and join them."""
    for env in self._envs:
      if hasattr(env, 'close'):
        env.close()
