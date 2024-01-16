from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.autograd import Variable

from ..agents.actors import ParamActor, QActor
from ..agents.agent import Agent
from ..agents.memory.memory import Memory
from ..agents.utils import hard_update_target_network, soft_update_target_network
from ..agents.utils.noise import OrnsteinUhlenbeckActionNoise

noise_std = 0.5


def constraint(act, act_param, constraint):
    """_summary_

    Args:
        act (_type_): _description_
        act_param (_type_): _description_
        constraint (_type_): _description_

    Returns:
        _type_: _description_
    """

    if act in constraint:

        if constraint[act][0] == "U":
            act_param = min(constraint[act][1], np.array(act_param))

        if constraint[act][0] == "L":
            act_param = max(constraint[act][1], np.array(act_param))

    return act, act_param


class PDQNAgent(Agent):
    """
    Similar style to DDPG, Actor(actor_param)+Critic(actor)

    """

    NAME = "P-DQN Agent"

    def __init__(
        self,
        cfg: DictConfig,
        observation_dim,
        action_dim,
        action_high,
        action_low,
        actor_class=QActor,
        actor_kwargs={},
        actor_param_class=ParamActor,
        actor_param_kwargs={},
        loss_func=F.mse_loss,  # F.mse_loss
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):

        super(PDQNAgent, self).__init__(
            observation_dim, action_dim, action_high, action_low
        )

        self.cfg = cfg

        self.device = torch.device(device)
        self.num_actions = self.action_dim
        self.action_parameter_sizes = np.ones(self.num_actions)
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = (
            torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        )
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()
        self.action_parameter_max_numpy = self.action_high.ravel()
        self.action_parameter_min_numpy = self.action_low.ravel()
        self.action_parameter_range_numpy = (
            self.action_parameter_max_numpy - self.action_parameter_min_numpy
        )
        self.action_parameter_max = (
            torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        )
        self.action_parameter_min = (
            torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        )
        self.action_parameter_range = (
            torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        )

        assert (
            cfg.agent.weighted ^ cfg.agent.average ^ cfg.agent.random_weighted
        ) or not (cfg.agent.weighted or cfg.agent.average or cfg.agent.random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self._step = 0
        self._episode = 0
        self.updates = 0

        self.np_random = np.random

        self.noise = OrnsteinUhlenbeckActionNoise(
            self.action_parameter_size,
            random_machine=self.np_random,
            mu=1.0,
            theta=3,
            sigma=0.0001,
        )  # , theta=0.01, sigma=0.01)
        self.trainpurpose = True

        self.replay_memory = Memory(
            cfg.agent.replay_memory_size,
            self.observation_dim,
            1 + self.action_parameter_size,
            next_actions=False,
        )
        self.actor = actor_class(
            self.observation_dim,
            self.num_actions,
            self.action_parameter_size,
            **actor_kwargs
        ).to(device)
        self.actor_target = actor_class(
            self.observation_dim,
            self.num_actions,
            self.action_parameter_size,
            **actor_kwargs
        ).to(device)

        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(
            self.observation_dim,
            self.num_actions,
            self.action_parameter_size,
            **actor_param_kwargs
        ).to(device)
        self.actor_param_target = actor_param_class(
            self.observation_dim,
            self.num_actions,
            self.action_parameter_size,
            **actor_param_kwargs
        ).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = (
            loss_func  # l1_smooth_loss performs better but original paper used MSE
        )

        self.actor_optimiser = optim.Adam(
            self.actor.parameters(), lr=cfg.agent.learning_rate_actor
        )
        self.actor_param_optimiser = optim.Adam(
            self.actor_param.parameters(), lr=cfg.agent.learning_rate_actor
        )

        self.epsilon = cfg.agent.epsilon_initial

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += (
            "Actor Network {}\n".format(self.actor)
            + "Param Network {}\n".format(self.actor_param)
            + "Actor Alpha: {}\n".format(self.learning_rate_actor)
            + "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param)
            + "Gamma: {}\n".format(self.gamma)
            + "Tau (actor): {}\n".format(self.tau_actor)
            + "Tau (actor-params): {}\n".format(self.tau_actor_param)
            + "Inverting Gradients: {}\n".format(self.inverting_gradients)
            + "Replay Memory: {}\n".format(self.replay_memory_size)
            + "Batch Size: {}\n".format(self.batch_size)
            + "Initial memory: {}\n".format(self.initial_memory_threshold)
            + "epsilon_initial: {}\n".format(self.epsilon_initial)
            + "epsilon_final: {}\n".format(self.epsilon_final)
            + "epsilon_steps: {}\n".format(self.epsilon_steps)
            + "Clip Grad: {}\n".format(self.clip_grad)
            + "Ornstein Noise?: {}\n".format(self.use_ornstein_noise)
            + "Zero Index Grads?: {}\n".format(self.zero_index_gradients)
            + "Seed: {}\n".format(self.seed)
        )
        return desc

    def set_action_parameter_passthrough_weights(
        self, initial_weights, initial_bias=None
    ):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer

        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = (
            torch.Tensor(initial_weights).float().to(self.device)
        )
        if initial_bias is not None:

            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = (
                torch.Tensor(initial_bias).float().to(self.device)
            )
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """Continuous action exploration using an Ornstein–Uhlenbeck process."""
        return all_action_parameters.data.numpy() + (
            self.noise.sample() * self.action_parameter_range_numpy
        )

    def start_episode(self):
        pass

    def end_episode(self):

        self._episode += 1
        ep = self._episode
        if ep < self.cfg.agent.epsilon_steps:
            self.epsilon = self.cfg.agent.epsilon_initial - (
                self.cfg.agent.epsilon_initial - self.cfg.agent.epsilon_final
            ) * (ep / self.cfg.agent.epsilon_steps)
        else:
            self.epsilon = self.cfg.agent.epsilon_final

    def act(self, state, action_available):
        """
        Get the next action from the agent

        """

        with torch.no_grad():

            all_action_parameters = self.actor_param.forward(state)

            # Exploration
            if self.np_random.uniform() < self.epsilon:
                action = action_available[self.np_random.choice(len(action_available))]
                if not self.cfg.agent.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(
                        np.random.uniform(
                            self.action_parameter_min_numpy,
                            self.action_parameter_max_numpy,
                        )
                    )

            # Exploitation
            else:
                Q_a = self.actor.forward(
                    state.unsqueeze(0), all_action_parameters.unsqueeze(0)
                )
                Q_a = Q_a.detach().cpu().data.numpy()
                Q_a = Q_a[0, action_available]
                action = action_available[np.argmax(Q_a)]

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array(
                [self.action_parameter_sizes[i] for i in range(action)], dtype=int
            ).sum()

            if self.cfg.agent.use_ornstein_noise and self.noise is not None:
                all_action_parameters[
                    offset : offset + int(self.action_parameter_sizes[action])
                ] += self.noise.sample()[
                    offset : offset + int(self.action_parameter_sizes[action])
                ]

            if self.cfg.agent.norm_noise and self.trainpurpose:
                if self.np_random.uniform() < self.epsilon:
                    if action == 0:

                        all_action_parameters[offset : offset + 1] += np.random.normal(
                            0, self.cfg.agent.noise_std, 1
                        )

            action_parameters = all_action_parameters[offset : offset + 1]
            action, action_parameters = constraint(
                action, action_parameters, self.cfg.general.rule
            )

        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):

        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()

        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[
                    self.action_parameter_offsets[a] : self.action_parameter_offsets[
                        a + 1
                    ]
                ] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.0
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):

        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()

        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def step(
        self, state, action, reward, next_state, next_action, terminal, time_steps=1
    ):
        """ """

        act, all_action_parameters = action
        self._step += 1
        action_to_store = np.concatenate(([act], all_action_parameters)).ravel()
        next_action_to_store = np.concatenate(
            ([next_action[0]], next_action[1])
        ).ravel()

        self._add_sample(
            state,
            action_to_store,
            reward,
            next_state,
            next_action_to_store,
            terminal=terminal,
        )

        if (
            self._step >= self.cfg.agent.batch_size
            and self._step >= self.cfg.agent.initial_memory_threshold
        ):
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        """
        Add a sample to the replay memory

        """

        assert len(action) == 1 + self.action_parameter_size
        actions = int(action[0])
        action_parameters = torch.from_numpy(action[1:]).float().to(self.device)
        Q_a = self.actor.forward(
            state.unsqueeze(0), action_parameters.unsqueeze(0)
        ).data
        Q_a_target = self.actor_target.forward(
            state.unsqueeze(0), action_parameters.unsqueeze(0)
        ).data

        t = (
            reward
            if terminal
            else reward + self.cfg.agent.gamma * torch.max(Q_a_target)
        )
        error = (
            (abs(Q_a[0][actions] - t).data.cpu().numpy()) + 0.01
        ) ** self.cfg.agent.priority

        self.replay_memory.append(
            error, state, action, reward, next_state, terminal=terminal
        )

    def _optimize_td_loss(self):

        if (
            self._step < self.cfg.agent.batch_size
            or self._step < self.cfg.agent.initial_memory_threshold
        ):
            return

        # Sample a batch
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(
            self.cfg.agent.batch_size, random_machine=self.np_random
        )

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(
            self.device
        )  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(
                next_states.to(torch.float64)
            )
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # TD error
            target = rewards + (1 - terminals) * self.cfg.agent.gamma * Qprime

        # Compute Q-values
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.cfg.agent.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.agent.clip_grad
            )
        self.actor_optimiser.step()

        # ---------------------- optimize pi ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        assert (
            self.cfg.agent.weighted
            ^ self.cfg.agent.average
            ^ self.cfg.agent.random_weighted
        ) or not (
            self.cfg.agent.weighted
            or self.cfg.agent.average
            or self.cfg.agent.random_weighted
        )
        Q = self.actor(states, action_params)
        Q_val = Q
        Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad()
        Q_loss.backward()

        from copy import deepcopy

        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(
            delta_a, action_params, grad_type="action_parameters", inplace=True
        )
        if self.cfg.agent.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(
                delta_a, batch_action_indices=actions, inplace=True
            )

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        if self.cfg.agent.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_param.parameters(), self.cfg.agent.clip_grad
            )

        self.actor_param_optimiser.step()

        soft_update_target_network(
            self.actor, self.actor_target, self.cfg.agent.tau_actor
        )
        soft_update_target_network(
            self.actor_param, self.actor_param_target, self.cfg.agent.tau_actor_param
        )

    def save_models(self, prefix):

        torch.save(self.actor.state_dict(), prefix + "_actor.pt")
        torch.save(self.actor_param.state_dict(), prefix + "_actor_param.pt")
        print("Models saved successfully")

    def load_models(self, prefix):

        self.actor.load_state_dict(torch.load(prefix + "_actor.pt", map_location="cpu"))
        self.actor_param.load_state_dict(
            torch.load(prefix + "_actor_param.pt", map_location="cpu")
        )
        print("Models loaded successfully")

    def copy_models(self, actor, actor_param):

        self.actor = deepcopy(actor)
        self.actor_param = deepcopy(actor_param)
