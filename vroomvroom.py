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

from libvroom import PPO, ProximityReward, NoMovementPenalty, merge_models

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
        reward_fn=ProximityReward(),#CombinedReward(
        #     (GoalReward(), 10.),
        #     (TouchReward(), 1.),
        #     (ProximityReward(), .1)
        # ),
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

    while not terminated and not truncated:
        if render:
            training_env.render()
            time.sleep(6/120)

        for agent_id, action_space in training_env.action_spaces.items():
            states[agent_id].append(obs_dict[agent_id])
            actions[agent_id].append(ppo.act(obs_dict[agent_id]))

        obs_dict, reward_dict, terminated_dict, truncated_dict = training_env.step({agent_id: actions[agent_id][-1] for agent_id in training_env.agents})

        for agent in reward_dict.keys():
            rewards[agent].append(reward_dict[agent])
            dones[agent].append(terminated_dict[agent] or truncated_dict[agent])
            next_states[agent].append(obs_dict[agent])

        truncated = True in list(truncated_dict.values())
        terminated = True in list(terminated_dict.values())


        if any(chain(terminated_dict.values(), truncated_dict.values())):
            break
    if not queue:
        return states, actions, rewards, next_states, dones
    else:
        queue.put((states, actions, rewards, next_states, dones))
        return None


if threads == 1:
    for _ in tqdm(range(passes)):
        states, actions, rewards, next_states, dones = simulate(agent, getenv, render)
        for agent_id in list(states.keys()):
            agent.learn(states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], dones[agent_id])
        if _ % 1000 == 0:
            agent.save_model(f"models/{now}.pth")
else:

    is_frozen = False


    progress_bar = tqdm(total=passes)
    def simulate_wrapper(i, queue):
        return simulate(agent, getenv, False, queue=queue)
    for i in range(passes//threads):
        queues = [multiprocessing.Queue() for _ in range(threads)]
        ps = []
        for i in range(threads):
            p = multiprocessing.Process(target=simulate_wrapper, args=(i, queues[i]))
            p.start()
            ps.append(p)
        finished = []
        done = False
        def update_finished():
            global finished,done
            while len(finished) != len(queues):
                for i in range(len(queues)):
                    if not queues[i].empty() and i not in finished:
                        finished.append(i)
            done = True
        threading.Thread(target=update_finished).start()
        jpm = 0
        while not done or len(finished) != 0:
            # bandaid code to prevent infinite loop
            jpm+=1
            if jpm == 10000000:
                print("Infinite loop detected, breaking")
                if len(finished) == 0:
                    finished = [i.get() for i in queues]
            if len(finished) == 0:
                pass
            else:
                result = queues[finished[0]].get()
                for agent_id in list(result[0].keys()):
                    agent.learn(result[0][agent_id], result[1][agent_id], result[2][agent_id], result[3][agent_id], result[4][agent_id])
                progress_bar.update(1)
                finished.pop(0)
        if i % 1000//threads == 0:
            agent.save_model(f"models/{now}.pth")




    agent.save_model(f"models/{now}.pth")


env.close()
