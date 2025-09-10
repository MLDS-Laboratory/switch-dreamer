import functools

import elements
import embodied
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

class RiskyCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = obs[0]
        violation = x_position > 0.01
        if violation:
            reward += 10.0 * np.random.randn()
        info['is_violation'] = violation
        return obs, reward, done, truncated, info
    
class RiskyInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        x_position = obs[0]
        violation = x_position > 0.01
        if violation:
            reward += 10.0 * np.random.randn()
        info['is_violation'] = violation
        return obs, reward, done, truncated, info
    
class RiskySwimmerEnv(SwimmerEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position > 0.5
        if violation:
            reward += 10.0 * np.random.randn()
        info['is_violation'] = violation
        return obs, reward, terminated, truncated, info
    
class RiskyHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        x_position = info['x_position']
        violation = x_position < -3
        if violation:
            reward += 10.0 * np.random.randn()
        info['is_violation'] = violation
        return obs, reward, terminated, truncated, info
    
register(
    id="RiskyCartPole-v0",
    entry_point=RiskyCartPoleEnv
)
register(
    id="RiskySwimmer-v0",
    entry_point=RiskySwimmerEnv
)
register(
    id="RiskyHalfCheetah-v0",
    entry_point=RiskyHalfCheetahEnv
)
register(
    id="RiskyInvertedPendulum-v0",
    entry_point=RiskyInvertedPendulumEnv
)

class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._truncated = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/is_violation': elements.Space(bool)
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, self._done, self._truncated, self._info = self._env.step(action)
    self._done = self._done or self._truncated
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)),
        is_violation=bool(self._info.get('is_violation', False)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False, is_violation=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    obs.update({'log/is_violation': is_violation})
    return obs

  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)
