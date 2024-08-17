import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
torch.multiprocessing.set_sharing_strategy('file_system')

# My implementation of the Proximal Policy Optimization (PPO) algorithm
# Serrano Academy: https://www.youtube.com/watch?v=TjHH_--7l8g
class PPO:
    def __init__(self, state_dim, action_dim,lr=1e-4, gamma=0.99, K=3, eps_clip=0.2, policysize=1024, valuesize=1024, device=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policynet = nn.Sequential(
            nn.Linear(state_dim, int(policysize/2)),
            nn.ReLU(),
            nn.Linear(int(policysize/2), policysize),
            nn.ReLU(),
            nn.Linear(policysize, int(policysize/2)),
            nn.ReLU(),
            nn.Linear(int(policysize/2), action_dim)
        ).to(self.device)

        self.valuenet = nn.Sequential(
            nn.Linear(state_dim, int(valuesize/2)),
            nn.ReLU(),
            nn.Linear(int(valuesize/2), valuesize),
            nn.ReLU(),
            nn.Linear(valuesize, int(valuesize/2)),
            nn.ReLU(),
            nn.Linear(int(valuesize/2), 1)
        ).to(self.device)

        self.optimizer = Adam(list(self.policynet.parameters()) + list(self.valuenet.parameters()), lr=lr)

        self.loss = nn.MSELoss()

        self.lock = False

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.policynet(state)
        dist = torch.distributions.Categorical(logits=action_probs)
        action = dist.sample().cpu().detach()
        log_probs = dist.log_prob(action).detach()
        action = action.numpy()
        return np.array([action]).astype(np.int32), log_probs

    def learn(self, states, actions, rewards, next_states, dones, log_probs):
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.tensor(np.array(actions)).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(log_probs).to(self.device)
        # GAE parameters
        lambda_ = 0.95

        with torch.no_grad():
            values = self.valuenet(states).squeeze()
            next_values = self.valuenet(next_states).squeeze()
            # GAE
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * lambda_ * (1 - dones[t]) * gae
                advantages[t] = gae

            # Compute returns from advantages
            returns = advantages + values

        # Normalizing the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K):
            self.optimizer.zero_grad()

            # Policy loss with PPO clipping
            action_probs = self.policynet(states)
            dist = torch.distributions.Categorical(logits=action_probs)
            old_log_probs = log_probs
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = self.loss(self.valuenet(states).squeeze(), returns)

            # Entropy for exploration
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy  # 0.5 is a common coefficient for value loss

            # Compute gradient and apply gradient clipping
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(list(self.policynet.parameters()) + list(self.valuenet.parameters()), max_norm=0.5)

            self.optimizer.step()
    def save_model(self, path):
        torch.save({
            'policynet_state_dict': self.policynet.state_dict(),
            'valuenet_state_dict': self.valuenet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policynet.load_state_dict(checkpoint['policynet_state_dict'])
        self.valuenet.load_state_dict(checkpoint['valuenet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def merge_models(models: List[PPO]):
    policies = [model.policynet.parameters() for model in models]
    values = [model.valuenet.parameters() for model in models]

    policyparams = [p.data for p in policies[0]]
    valueparams = [v.data for v in values[0]]

    for i in range(1, len(policies)):
        for j, p in enumerate(policies[i]):
            policyparams[j] += p.data
        for j, v in enumerate(values[i]):
            valueparams[j] += v.data

    for i in range(len(policyparams)):
        policyparams[i] /= len(policies)
    for i in range(len(valueparams)):
        valueparams[i] /= len(values)

    newmodel = PPO(policyparams[0].shape[0], models[0].policynet[0].out_features, models[0].valuenet[0].out_features, gamma=models[0].gamma, eps_clip=models[0].eps_clip, K=models[0].K, device=models[0].device)

    for oldp, newp in zip(newmodel.policynet.parameters(), policyparams):
        oldp.data = newp

    for oldv, newv in zip(newmodel.valuenet.parameters(), valueparams):
        oldv.data = newv

    return newmodel


class ProximityReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        carpos = state.cars[agent].physics.position
        ballpos = state.ball.position
        distance = np.linalg.norm([carpos[0] - ballpos[0], carpos[1] - ballpos[1], carpos[2] - ballpos[2]])
        reward = 1/(distance+1)
        return float(reward)


class NoMovementPenalty(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self.lastpos = {}
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for agent in agents:
            self.lastpos[agent] = initial_state.cars[agent].physics.position
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        self.lastpos[agent] = state.cars[agent].physics.position
        return 1 if list(self.lastpos[agent].astype(np.int32)) != list(state.cars[agent].physics.position.astype(np.int32)) else -1



class SpeedReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        return int(np.linalg.norm(state.cars[agent].physics.linear_velocity))

class BallPosReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        ballpos = state.ball.position
        if str(agent).startswith("blue"):
            return 1 if ballpos[1] < 0 else -1
        else:
            return 1 if ballpos[1] > 0 else -1
        return 0
