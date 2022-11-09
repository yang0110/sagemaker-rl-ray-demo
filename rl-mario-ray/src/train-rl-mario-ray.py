import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from ray.rllib.env.wrappers.atari_wrappers import (MonitorEnv,
                                          NoopResetEnv,
                                          WarpFrame,
                                          FrameStack)
import json
import os

import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher

class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        if info['time']<300:
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs

    
class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']    
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info


def create_environment(env_config):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = CustomReward(env)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env, 84)
    env = FrameStack(env, 4)    
    return env 

class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        register_env('SuperMarioBros-v0', create_environment)

    def get_experiment_config(self):
        return {
            "training": {
                "env": 'SuperMarioBros-v0',
                "run": "IMPALA",
                "stop": {"training_iteration": 30},
                "config": {
                    "framework": "torch",
                    "model": {"free_log_std": True},
                    "num_workers": (self.num_cpus -1),
                    "num_gpus": self.num_gpus,
                    'num_envs_per_worker': 1,
                    "batch_mode": "truncate_episodes",
                    'train_batch_size': 5000,
                },
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()