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

import functools

import tensorflow as tf

from planet import control
from planet.control.simulate import define_batch_env
from planet.control.simulate import simulate_step

from planet import tools
from planet.training import define_summaries
from planet.training import utility


def define_model(data, trainer, config):
  print("\nI'M INSIDE DEFINE MODEL\n")
  #with tf.device('/gpu:1'):
  tf.logging.info('Build TensorFlow compute graph.')
  dependencies = []
  cleanups = []
  step = trainer.step
  global_step = trainer.global_step
  phase = trainer.phase

  # Instantiate network blocks.
  cell = config.cell()
  kwargs = dict(create_scope_now_=True)
  encoder = tf.make_template('encoder', config.encoder, **kwargs)
  heads = tools.AttrDict(_unlocked=True)
  dummy_features = cell.features_from_state(cell.zero_state(1, tf.float32))
  for key, head in config.heads.items():
    name = 'head_{}'.format(key)
    kwargs = dict(create_scope_now_=True)
    if key in data:
      kwargs['data_shape'] = data[key].shape[2:].as_list()
    elif key == 'action_target':
      kwargs['data_shape'] = data['action'].shape[2:].as_list()
    heads[key] = tf.make_template(name, head, **kwargs)
    heads[key](dummy_features)  # Initialize weights.

  # Apply and optimize model.
  embedded = encoder(data)
  with tf.control_dependencies(dependencies):
    embedded = tf.identity(embedded)
  graph = tools.AttrDict(locals())
  prior, posterior = tools.unroll.closed_loop(
      cell, embedded, data['action'], config.debug)
  objectives = utility.compute_objectives(
      posterior, prior, data, graph, config)
  summaries, grad_norms = utility.apply_optimizers(
      objectives, trainer, config)

  ## CREATE THE ENVIRONMENT AND PASS IT IN EVERYWHERE ##
  print("CREATING THE IN GRAPH BATCH ENV")
  def define_model_env_ctor():
    env = params.task.env_ctor()
    if params.save_episode_dir:
      env = control.wrappers.CollectGymDataset(env, params.save_episode_dir)
      #env = control.wrappers.CollectGymDataset.get_my_env(env, params.save_episode_dir)
    env = control.wrappers.ConcatObservation(env, ['image'])
    #env = control.wrappers.ConcatObservation.get_my_env(env, ['image'])
    return env
  params_list = list(config.train_collects.items())   # [(name, params)]
  params = params_list[0][1]  # Only should be 1 element in this list
  ingraphbatchenv = define_batch_env(define_model_env_ctor, params.num_agents, config.isolate_envs)
  print("LOOKING AT ONION OF WRAPPERS")
  l1 = ingraphbatchenv._batch_env
  print(l1)
  l2 = l1._envs
  print(l2)  # ConcatObservation
  l3 = l2[0]._env
  print(l3) # CollectGymDataset
  l4 = l3._env 
  print(l4) # MaximumDuration
  l5 = l4._env
  print(l5) # ActionRepeat
  l6 = l5._env
  print(l6) # Stanford

  ## CALL SIMULATE_STEP AND PASS THE OUTPUT EVERYWHERE ##
  bind_or_none = lambda x, **kw: x and functools.partial(x, **kw)
  celll = graph.cell
  agent_config = tools.AttrDict(
      cell=celll,
      encoder=graph.encoder,
      planner=functools.partial(params.planner, graph=graph),
      objective=bind_or_none(params.objective, graph=graph),
      exploration=params.exploration,
      preprocess_fn=config.preprocess_fn,
      postprocess_fn=config.postprocess_fn)
  stepp = graph.step
  agent = control.mpc_agent.MPCAgent(ingraphbatchenv, stepp, False, False, agent_config)
  donee, scoree, unused_summary = simulate_step(
       ingraphbatchenv, agent,
       log=False,
       reset=tf.equal(stepp, 0))
  print("DONEE, SCOREE", donee, scoree)
  # donee = tf.zeros([params.num_agents], tf.bool)
  # scoree = tf.zeros([params.num_agents], tf.float32)


  # outside_summary = None
  # outside_return = None
  
  # Active data collection.
  with tf.variable_scope('collection'):
    with tf.control_dependencies(summaries):  # Make sure to train first.
      for name, params in config.train_collects.items():
        #print("INSIDE DEFINE MODEL DATA COLLECTION", name, params)
        schedule = tools.schedule.binary(
            step, config.batch_shape[0],
            params.steps_after, params.steps_every, params.steps_until)
        # summary, _ = tf.cond(
        #     tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
        #     functools.partial(
        #         utility.simulate_episodes, config, params, graph, cleanups,
        #         expensive_summaries=False, gif_summary=False, name=name),
        #     lambda: (tf.constant(''), tf.constant(0.0)),
        #     name='should_collect_' + name)
        summary, _ = tf.cond(
            tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
            functools.partial(
                utility.simulate_episodes, config, params, graph, cleanups,
                expensive_summaries=False, gif_summary=False, name=name, batchenv=ingraphbatchenv,
                donee=donee, scoree=scoree),
            lambda: (tf.constant(''), tf.constant(0.0)),
            name='should_collect_' + name)
        # ssummary, ret = tf.cond(
        #     tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
        #     functools.partial(
        #         utility.simulate_episodes, config, params, graph, cleanups,
        #         expensive_summaries=False, gif_summary=False, name=name),
        #     lambda: (tf.constant(''), tf.constant(0.0)),
        #     name='should_collect_' + name)
        summaries.append(summary)
        #summaries.append(ssummary)

        # outside_summary = ssummary
        # outside_return = ret

  # print("SSUMARY, RET", outside_summary, outside_return)

  # Compute summaries.
  graph = tools.AttrDict(locals())
  print("trainer.log", trainer.log)
  # summary, score = tf.cond(
  #     trainer.log,
  #     lambda: define_summaries.define_summaries(graph, config, cleanups),
  #     lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
  #     name='summaries')
  summary, score = tf.cond(
      trainer.log,
      lambda: define_summaries.define_summaries(graph, config, cleanups, batchenv=ingraphbatchenv,
                                                donee=donee, scoree=scoree),
      lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
      name='summaries')
  summaries = tf.summary.merge([summaries, summary])
  dependencies.append(utility.print_metrics(
      {ob.name: ob.value for ob in objectives},
      step, config.print_metrics_every, 'objectives'))
  dependencies.append(utility.print_metrics(
      grad_norms, step, config.print_metrics_every, 'grad_norms'))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)
  return score, summaries, cleanups
