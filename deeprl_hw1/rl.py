# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np

max_int = 100000;
epsilon = 1e-3

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    delta = max_int;
    V = np.zeros(env.nS);
    cnt = 0;

    while (delta > tol and cnt < max_iterations):
      delta = 0;
      for s in range(env.nS):
        v = 0;
        a = policy[s];
        for prob, next_state, reward, terminal in env.P[s][a]:
          v += prob * (reward + gamma * V[next_state]);
        delta = max(delta, abs(v-V[s]));
        V[s] = v;
        cnt += 1;

    return V, cnt


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    

    policy = np.zeros(env.nS, dtype='int');

    for s in range(env.nS):
      v_max = -max_int;
      a_max = 0;
      for a in range(env.nA):
          v = 0;
          for prob, next_state, reward, terminal in env.P[s][a]:
            v += prob * (reward + gamma * value_function[next_state]);
          if (v > v_max):
            v_max = v;
            a_max = a;
          if (s==2):
            print(v);
      policy[s] = a_max;
    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    policy_ = value_function_to_policy(env,gamma,value_func)
    flag = False;

    for s in range(env.nS):
      for a in range(env.nA):
        if (policy_[s] != policy[s]):
          flag = True;
          break;
      if (flag): break;

    return flag, policy_


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """

    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    cnt = 0;
    val_cnt = 0;
    flag = True;

    while (flag and cnt < max_iterations):
      value_func, v_cnt = evaluate_policy(env, gamma, policy)
      val_cnt += v_cnt;
      flag, policy = improve_policy(env, gamma, value_func, policy);
      cnt += 1;
    return policy, value_func, cnt, val_cnt


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    V = np.zeros(env.nS);
    cnt = 0;
    delta = max_int;

    while (delta > tol and cnt < max_iterations):
      delta = 0;
      for s in range(env.nS):
        v_max = - max_int;
        for a in range(env.nA):
          v = 0;
          for prob, next_state, reward, terminal in env.P[s][a]:
            v += prob * (reward + gamma * V[next_state]);
          v_max = max(v, v_max);
        delta = max(delta, abs(v - V[s]));
        V[s] = v_max;
      cnt += 1;
    return V, cnt;


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
