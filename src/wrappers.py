import atexit
import functools
import sys
import threading
import traceback
import os

import gym
import numpy as np
from PIL import Image
import collections

os.environ['MUJOCO_GL'] = 'egl'

from dm_control.suite import cheetah
from collections import namedtuple

Task = collections.namedtuple('Task', 'name, env_ctor, state_components')
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None, zero_shot=False):
    domain, task = name.split('_', 1)
    if domain == 'cup':
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      if zero_shot:
        from dm_control.suite import cheetah
        cheetah._RUN_SPEED = 15
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    # spaces = {}
    # for key, value in self._env.observation_spec().items():
    #   spaces[key] = gym.spaces.Box(
    #       -np.inf, np.inf, value.shape, dtype=np.float32)
    # spaces['image'] = gym.spaces.Box(
    #     0, 255, self._size + (3,), dtype=np.uint8)
    # return gym.spaces.Dict(spaces)
    return gym.spaces.Box(0, 255, self._size+(3, ), dtype=np.uint8)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, size=None,  *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    if size is None:
      return self._env.physics.render(*self._size, camera_id=self._camera)
    return self._env.physics.render(*size, camera_id=self._camera)

  @property
  def spaces(self):
    return EnvSpaces(
        observation=self.observation_space,
        action=self.action_space,
    )

class cheetah_run_back:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cheetah' and task == 'run': 
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      raise NotImplementedError('Back run currently only support cheetah')
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    speed = -self._env.physics.speed() or 0
    reward = max(0, min(speed/10, 1))
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, size=None,  *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    if size is None:
      return self._env.physics.render(*self._size, camera_id=self._camera)
    return self._env.physics.render(*size, camera_id=self._camera)