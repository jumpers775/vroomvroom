import time
import numpy as np
from itertools import chain
from scipy.interpolate import interp1d

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator

from libvroom import PPO, ProximityReward, NoMovementPenalty, merge_models, BallPosReward, SpeedReward

import torch
from tqdm import tqdm
import os
import datetime
import sys
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
import multiprocessing
import threading

parser = argparse.ArgumentParser(
                    prog='vroomvroom',
                    description='rocket league bot management',
                    epilog='vroom vroom')

parser.add_argument('-m', '--model', type=str, help='model to load', default=None)
parser.add_argument('-r', '--render', action='store_true', help='render the game', default=False)
parser.add_argument('-p', '--passes', type=int, help='number of passes', default=int(1e6))
parser.add_argument('-t', '--threads', type=int, help='number of threads', default=os.cpu_count())

args = parser.parse_args()

render = args.render

passes = args.passes - (args.passes % args.threads)

threads = args.threads if not render else 1

def getenv():
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2),
            KickoffMutator()
        ),
        obs_builder=DefaultObs(zero_padding=2),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=CombinedReward(
            (GoalReward(), 12.),
            (TouchReward(), 3.),
            (ProximityReward(), 1.),
            (BallPosReward(), 2.),
            (SpeedReward(), 0.5)
        ),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(timeout=300.),
            NoTouchTimeoutCondition(timeout=30.)
        ),
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )

env = getenv()
obs_dict = env.reset()

obs_space_dims = env.observation_space(env.agents[0])[1]

action_space_dims = env.action_space(env.agents[0])[1]

agent = PPO(obs_space_dims, action_space_dims, device=torch.device("cpu")) # cpu is required for multiprocessing, which is faster than cuda without it

print("Using device: ", agent.device)

if not os.path.exists("models"):
    os.mkdir("models")
models = os.listdir("models")

if args.model:
    agent.load_model(args.model)



losses = []

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def simulate(ppo: PPO, env_builder, render: bool, queue = None):
    training_env = env_builder()

    obs_dict = training_env.reset()

    terminated = False
    truncated = False

    states = {agent: [] for agent in training_env.agents}
    actions = {agent: [] for agent in training_env.agents}
    rewards = {agent: [] for agent in training_env.agents}
    next_states = {agent: [] for agent in training_env.agents}
    dones = {agent: [] for agent in training_env.agents}
    log_probs = {agent: [] for agent in training_env.agents}

    while not terminated and not truncated:
        if render:
            training_env.render()
            time.sleep(6/120)

        for agent_id, action_space in training_env.action_spaces.items():
            states[agent_id].append(obs_dict[agent_id])
            action, logprobs = ppo.act(obs_dict[agent_id])
            actions[agent_id].append(action)
            log_probs[agent_id].append(logprobs)

        obs_dict, reward_dict, terminated_dict, truncated_dict = training_env.step({agent_id: actions[agent_id][-1] for agent_id in training_env.agents})

        for agent in reward_dict.keys():
            rewards[agent].append(reward_dict[agent])
            dones[agent].append(terminated_dict[agent] or truncated_dict[agent])
            next_states[agent].append(obs_dict[agent])

        truncated = True in list(truncated_dict.values())
        terminated = True in list(terminated_dict.values())


        if any(chain(terminated_dict.values(), truncated_dict.values())):
            break
    return states, actions, rewards, next_states, dones, log_probs


if threads == 1:
    for _ in tqdm(range(passes)):
        states, actions, rewards, next_states, dones, log_probs = simulate(agent, getenv, render)
        for agent_id in list(states.keys()):
            agent.learn(states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], dones[agent_id], log_probs[agent_id])
        if _ % 1000 == 0:
            agent.save_model(f"models/{now}.pth")
else:



    progress_bar = tqdm(total=passes)
    def simulate_wrapper(i):
        return simulate(agent, getenv, False)
    with ProcessPoolExecutor(max_workers=threads) as executor:
        for i in range(0, passes, threads):
            futures = {executor.submit(simulate_wrapper, i + j) for j in range(threads)}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
                progress_bar.update(1)
            for result in results:
                for agent_id in list(result[0].keys()):
                    agent.learn(result[0][agent_id], result[1][agent_id], result[2][agent_id], result[3][agent_id], result[4][agent_id], result[5][agent_id])


            if i % 1000//threads == 0:
                agent.save_model(f"models/{now}.pth")




    agent.save_model(f"models/{now}.pth")


env.close()
