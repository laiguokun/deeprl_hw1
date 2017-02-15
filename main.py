#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.lake_envs as lake_env
import deeprl_hw1.rl as rl
import gym
from gym import wrappers
import time
import numpy as np


def run_my_policy(env, policy):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    
    total_reward = 0
    num_steps = 0
    nextstate = initial_state;
    while True:
        action = policy[nextstate];
        nextstate, reward, is_terminal, debug_info = env.step(action)
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))

def upload(policy):
    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1',force=True)
    for i_episode in range(2000):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = policy[observation]
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    gym.upload('/tmp/frozenlake-experiment-1', api_key='sk_0Z6MMPCTgiAGwmwJ54zLQ')

def main():
    # create the environment
    # env = gym.make('FrozenLake-v0')
    # uncomment next line to try the deterministic version
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')

    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    #policy iteration
    policy, value_func, cnt1, cnt2 = rl.policy_iteration(env,0.9)
    #print(policy,value_func);
    #print(cnt1,cnt2);
    rl.print_policy(policy,lake_env.action_names);
    #upload(policy);

    #value_iteration
    value_func, cnt1 = rl.value_iteration(env, 0.9);
    policy = rl.value_function_to_policy(env, 0.9, value_func)
    rl.print_policy(policy,lake_env.action_names);
    print(value_func)
    #run_my_policy(env, policy)
    #print('Agent received total reward of: %f' % total_reward)
    #print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
