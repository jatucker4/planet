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

"""In-graph simulation step of a vectorized algorithm with environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import time

from planet import tools
from planet.control import batch_env
from planet.control import in_graph_batch_env
from planet.control import mpc_agent
from planet.control import wrappers
from planet.plotting import stanford_viz
from planet.tools import streaming_mean

from planet.control.batch_env import BatchEnv


def simulate(
    step, env_ctor, duration, num_agents, agent_config,
    isolate_envs='none', expensive_summaries=False,
    gif_summary=True, name='simulate', outdir=''):

  summaries = []
  with tf.variable_scope(name):
    return_, image, action, reward, cleanup, step_time, reached_goal = collect_rollouts(
        step=step,
        env_ctor=env_ctor,
        duration=duration,
        num_agents=num_agents,
        agent_config=agent_config,
        isolate_envs=isolate_envs)

    # Find the average time it took to plan during the episode
    # Episode is defined as the steps it took to reach the goal
    print("STEP TIMES", step_time)
    print("REACHED GOAL", reached_goal)
    reached_goal_indices = tf.where(tf.squeeze(reached_goal))
    print("REACHED GOAL INDICES", reached_goal_indices.shape)
    if tf.size(reached_goal_indices) == 0:
      print("INSIDE IF")
      avg_step_time = tf.reduce_mean(step_time)
    else:
      print("INSIDE ELSE")
      first_reached_goal_index = 100 #reached_goal_indices[0][0]
      step_time = step_time[:first_reached_goal_index + 1]
      avg_step_time = tf.reduce_mean(step_time)
    #print("AVG STEP TIME", avg_step_time)
    summaries.append(tf.summary.scalar('sampada_step_time', avg_step_time))
    #stanford_viz.visualize_step_time(avg_step_time, outdir)
    #print("STANFORD VIZ VISUALIZE STEP TIME\n")

    return_mean = tf.reduce_mean(return_)
    summaries.append(tf.summary.scalar('return', return_mean))
    if expensive_summaries:
      summaries.append(tf.summary.histogram('return_hist', return_))
      summaries.append(tf.summary.histogram('reward_hist', reward))
      summaries.append(tf.summary.histogram('action_hist', action))
      summaries.append(tools.image_strip_summary(
          'image', image, max_length=duration))
    if gif_summary:
     summaries.append(tools.gif_summary(
         'animation', image, max_outputs=1, fps=20))
  summary = tf.summary.merge(summaries)
  return summary, return_mean, cleanup


def collect_rollouts(
    step, env_ctor, duration, num_agents, agent_config, isolate_envs):
  batch_env = define_batch_env(env_ctor, num_agents, isolate_envs)
  #batch_env = batchenv
  agent = mpc_agent.MPCAgent(batch_env, step, False, False, agent_config)
  #print("INSIDE COLLECT ROLLOUTS", step)
  cleanup = lambda: batch_env.close()

  def simulate_fn(unused_last, step):
    done, score, step_time, unused_summary = simulate_step(
       batch_env, agent,
       log=False,
       reset=tf.equal(step, 0))
    
    step_time_temp = tf.constant([step_time])
    print("STEP TIME SHAPE", step_time_temp.shape)
    step_time = tf.get_variable(
          'step_time', (1,), tf.float32,
          tf.constant_initializer(step_time), trainable=False)
    #step_time.assign(step_time_temp)

    with tf.control_dependencies([done, score]):
      image = batch_env.observ
      batch_action = batch_env.action
      batch_reward = batch_env.reward
      # Warning: Assumes a particular wrapper onion!!
      inner_env = batch_env._batch_env._envs[0]._env._env._env._env
      print("INNER ENV", inner_env)
      #reached_goal = inner_env.reached_goal
      reached_goal = tf.get_variable(
          'reached_goal', (1,), tf.bool,
          tf.constant_initializer(inner_env.reached_goal), trainable=False)
      #reached_goal.assign(tf.constant([inner_env.reached_goal]))
    return done, score, image, batch_action, batch_reward, step_time, reached_goal

  initializer = (
      tf.zeros([num_agents], tf.bool),
      tf.zeros([num_agents], tf.float32),
      0 * batch_env.observ,
      0 * batch_env.action,
      tf.zeros([num_agents], tf.float32),
      tf.zeros([num_agents], tf.float32),
      tf.zeros([num_agents], tf.bool))
  done, score, image, action, reward, step_time, reached_goal = tf.scan(
      simulate_fn, tf.range(duration),
      initializer, parallel_iterations=1)
  score = tf.boolean_mask(score, done)
  image = tf.transpose(image, [1, 0, 2, 3, 4])
  action = tf.transpose(action, [1, 0, 2])
  reward = tf.transpose(reward)
  return score, image, action, reward, cleanup, step_time, reached_goal 


def define_batch_env(env_ctor, num_agents, isolate_envs):
  #print("I SHOULD ONLY BE IN HERE ONCE")
  with tf.variable_scope('environments'):
    if isolate_envs == 'none':
      factory = lambda ctor: ctor()
      blocking = True
    elif isolate_envs == 'thread':
      factory = functools.partial(wrappers.Async, strategy='thread')
      #factory = functools.partial(wrappers.Async.get_my_env, strategy='thread')
      blocking = False
    elif isolate_envs == 'process':
      factory = functools.partial(wrappers.Async, strategy='process')
      #factory = functools.partial(wrappers.Async.get_my_env, strategy='process')
      blocking = False
    else:
      raise NotImplementedError(isolate_envs)
    envs = [factory(env_ctor) for _ in range(num_agents)]
    #envs = [env_ctor() for _ in range(num_agents)]

    #env = BatchEnv.get_my_env(envs, blocking)
    #env = BatchEnv.get_my_env(factory, env_ctor, num_agents, blocking)
    env = batch_env.BatchEnv(envs, blocking)
    
    env = in_graph_batch_env.InGraphBatchEnv(env)
    #env = in_graph_batch_env.InGraphBatchEnv.get_my_env(env)
    
  return env 


def simulate_step(batch_env, algo, log=True, reset=False):
  """Simulation step of a vectorized algorithm with in-graph environments.

  Integrates the operations implemented by the algorithm and the environments
  into a combined operation.

  Args:
    batch_env: In-graph batch environment.
    algo: Algorithm instance implementing required operations.
    log: Tensor indicating whether to compute and return summaries.
    reset: Tensor causing all environments to reset. 

  Returns:
    Tuple of tensors containing done flags for the current episodes, possibly
    intermediate scores for the episodes, and a summary tensor.
  """

  def _define_begin_episode(agent_indices):
    """Reset environments, intermediate scores and durations for new episodes.

    Args:
      agent_indices: Tensor containing batch indices starting an episode.

    Returns:
      Summary tensor, new score tensor, and new length tensor.
    """
    assert agent_indices.shape.ndims == 1
    zero_scores = tf.zeros_like(agent_indices, tf.float32)
    zero_durations = tf.zeros_like(agent_indices)
    update_score = tf.scatter_update(score_var, agent_indices, zero_scores)
    update_length = tf.scatter_update(
        length_var, agent_indices, zero_durations)
    reset_ops = [
        batch_env.reset(agent_indices), update_score, update_length]
    with tf.control_dependencies(reset_ops):
      return algo.begin_episode(agent_indices), update_score, update_length

  def _define_step():
    """Request actions from the algorithm and apply them to the environments.

    Increments the lengths of all episodes and increases their scores by the
    current reward. After stepping the environments, provides the full
    transition tuple to the algorithm.

    Returns:
      Summary tensor, new score tensor, and new length tensor.
    """
    prevob = batch_env.observ + 0  # Ensure a copy of the variable value.
    agent_indices = tf.range(len(batch_env))
    t0 = time.time()
    action, step_summary = algo.perform(agent_indices, prevob)
    t1 = time.time()
    step_time = t1 - t0
    print("STEP TIME T0 T1", step_time)
    #print("INSIDE DEFINE STEP", action)
    action.set_shape(batch_env.action.shape)
    with tf.control_dependencies([batch_env.step(action)]):
      #print("SCORE VAR", score_var)
      add_score = score_var.assign_add(batch_env.reward)
      inc_length = length_var.assign_add(tf.ones(len(batch_env), tf.int32))
    with tf.control_dependencies([add_score, inc_length]):
      agent_indices = tf.range(len(batch_env))
      experience_summary = algo.experience(
          agent_indices, prevob,
          batch_env.action,
          batch_env.reward,
          batch_env.done,
          batch_env.observ)
      summary = tf.summary.merge([step_summary, experience_summary])
    return summary, add_score, inc_length, step_time

  def _define_end_episode(agent_indices):
    """Notify the algorithm of ending episodes.

    Also updates the mean score and length counters used for summaries.

    Args:
      agent_indices: Tensor holding batch indices that end their episodes.

    Returns:
      Summary tensor.
    """
    assert agent_indices.shape.ndims == 1
    submit_score = mean_score.submit(tf.gather(score, agent_indices))
    submit_length = mean_length.submit(
        tf.cast(tf.gather(length, agent_indices), tf.float32))
    with tf.control_dependencies([submit_score, submit_length]):
      return algo.end_episode(agent_indices)

  def _define_summaries():
    """Reset the average score and duration, and return them as summary.

    Returns:
      Summary string.
    """
    score_summary = tf.cond(
        tf.logical_and(log, tf.cast(mean_score.count, tf.bool)),
        lambda: tf.summary.scalar('mean_score', mean_score.clear()), str)
    length_summary = tf.cond(
        tf.logical_and(log, tf.cast(mean_length.count, tf.bool)),
        lambda: tf.summary.scalar('mean_length', mean_length.clear()), str)
    return tf.summary.merge([score_summary, length_summary])

  with tf.name_scope('simulate'):
    log = tf.convert_to_tensor(log)
    reset = tf.convert_to_tensor(reset)
    with tf.variable_scope('simulate_temporary'):
      score_var = tf.get_variable(
          'score', (len(batch_env),), tf.float32,
          tf.constant_initializer(0),
          trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
      length_var = tf.get_variable(
          'length', (len(batch_env),), tf.int32,
          tf.constant_initializer(0),
          trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    mean_score = streaming_mean.StreamingMean((), tf.float32, 'mean_score')
    mean_length = streaming_mean.StreamingMean((), tf.float32, 'mean_length')
    agent_indices = tf.cond(
        reset,
        lambda: tf.range(len(batch_env)),
        lambda: tf.cast(tf.where(batch_env.done)[:, 0], tf.int32))
    begin_episode, score, length = tf.cond(
        tf.cast(tf.shape(agent_indices)[0], tf.bool),
        lambda: _define_begin_episode(agent_indices),
        lambda: (str(), score_var, length_var))
    with tf.control_dependencies([begin_episode]):
      #t0 = time.time()
      step, score, length, step_time = _define_step()
      #t1 = time.time()
      #step_time = t1 - t0
      #print("STEP TIME T0 T1", step_time)
    with tf.control_dependencies([step]):
      agent_indices = tf.cast(tf.where(batch_env.done)[:, 0], tf.int32)
      end_episode = tf.cond(
          tf.cast(tf.shape(agent_indices)[0], tf.bool),
          lambda: _define_end_episode(agent_indices), str)
    with tf.control_dependencies([end_episode]):
      summary = tf.summary.merge([
          _define_summaries(), begin_episode, step, end_episode])
    with tf.control_dependencies([summary]):
      score = 0.0 + score
      done = batch_env.done
    return done, score, step_time, summary
