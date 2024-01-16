""""
Original project done by Ziheng Chen Paper: ReLAX: Reinforcement Learning Agent eXplainer for Arbitrary Predictive Models
You can find the original source at https://github.com/Mewtwo1996/ReLAX

Refactored by Flavio Giorgi
"""
import numpy as np
import torch
from hydra import compose, initialize
from pandas import DataFrame
from tqdm import tqdm

from carla.recourse_methods.api.recourse_method import RecourseMethod

from .agents.pdqn_bound_pri import PDQNAgent
from .environments.env18 import Environment
from .utils import *


class Relax(RecourseMethod):
    def __init__(self, mlmodel):
        super().__init__(mlmodel)

        initialize(version_base=None, config_path="conf")
        self.cfg = compose("config.yaml")

    def get_counterfactuals(self, factuals: DataFrame):
        """
        Generates counterfactual examples for a given set of factual data points.

        This method utilizes a reinforcement learning approach, specifically using two instances of the PDQNAgent
        (Probabilistic Deep Q-Network Agent). It explores the data space to find counterfactuals that are close to
        the original data points but yield different outcomes when fed into the machine learning model.

        Parameters:
        - factuals (DataFrame): A pandas DataFrame containing the factual data points. The last column is assumed
        to be the target variable.

        Steps:
        1. Convert the factual data into a numpy array and separate features from the target variable.
        2. Initialize two PDQNAgent instances with configuration parameters.
        3. Create an environment for the agent to interact with, using the provided machine learning model and
        configuration settings.
        4. Iterate over the dataset, allowing the agent to explore and learn from the environment.
        5. In each iteration, the agent performs a series of actions to modify the input data point with the goal
        of changing its predicted outcome.
        6. Record the rewards and track the best counterfactual found for each data point.
        7. Adjust the learning rate of the agent if necessary based on performance.
        8. Print the counterfactual, original prediction, and the distance between the original and counterfactual
        data points.

        Returns:
        A list of tuples where the first element is the original instance and the second element is the counterfactual one
        Important: the best example found is always returned even if it is not a counterfactual!

        Note:
        - The method assumes that the last column of the input DataFrame is the target variable.
        - The configuration for the agents and the environment is retrieved from the 'self.cfg' attribute of the class.
        - The machine learning model used for predictions is accessed via 'self._mlmodel'.
        """

        dataset = factuals.to_numpy()[:, :-1]
        y_orig = factuals.to_numpy()[:, -1]

        total_reward = 0.0
        returns, label = [], []

        observation_dim = dataset.shape[1]
        action_dim = dataset.shape[1]
        action_high = np.amax(dataset, axis=0)
        action_low = np.amin(dataset, axis=0)

        agent = PDQNAgent(
            self.cfg,
            observation_dim,
            action_dim,
            action_high,
            action_low,
            actor_kwargs={
                "hidden_layers": self.cfg.agent.layers,
                "action_input_layer": self.cfg.agent.action_input_layer,
            },
            actor_param_kwargs={
                "hidden_layers": self.cfg.agent.layers,
                "squashing_function": False,
                "output_layer_init_std": 0.0001,
            },
        )

        agent_val = PDQNAgent(
            self.cfg,
            observation_dim,
            action_dim,
            action_high,
            action_low,
            actor_kwargs={
                "hidden_layers": self.cfg.agent.layers,
                "action_input_layer": self.cfg.agent.action_input_layer,
            },
            actor_param_kwargs={
                "hidden_layers": self.cfg.agent.layers,
                "squashing_function": False,
                "output_layer_init_std": 0.0001,
            },
        )

        environment = Environment(
            observation_dim,
            action_dim,
            self._mlmodel,
            self.cfg.agent.balance_factor,
            self.cfg.general.max_steps,
        )
        count = 0
        agent_val.epsilon = 0
        agent_val.trainpurpose = False
        environment.build_meta_local(dataset)
        previous_reward = float("-inf")
        x_best = np.zeros(dataset.shape[1])
        cf_list = []

        for index in range(dataset.shape[0]):
            x = dataset[index]

            for episode in tqdm(range(self.cfg.agent.episodes)):
                model_prediction = self._mlmodel.predict(x.reshape(1, -1))
                episode_reward, flag = 0, 0

                if np.random.uniform(0, 1) < 0.5:
                    x, y, flag, i = dataset[index, :], model_prediction, 1, index

                else:
                    x, y, i = (
                        np.random.randn(dataset.shape[1]) * 0.1 + dataset[index, :],
                        model_prediction,
                        0,
                    )

                state = torch.from_numpy(x)
                environment.reset_local(state, y, i)

                act, act_param, all_action_parameters = agent.act(
                    state, environment.actions_available
                )
                action = pad_action(act, act_param, action_dim)

                for _ in range(self.cfg.general.max_steps):

                    (next_state, steps), reward, terminal = environment.step(action)

                    if terminal and flag == 1:
                        next_state_np = next_state.numpy()

                        # Distance between the next state and the current sample
                        dist1 = np.sum(np.abs(next_state_np - dataset[index, :]))

                        # Distance between the best state and the current sample
                        dist2 = np.sum(np.abs(x_best - dataset[index, :]))

                        if dist1 < dist2:
                            x_best = next_state_np

                    next_act, next_act_param, next_all_action_parameters = agent.act(
                        next_state, environment.actions_available
                    )
                    next_action = pad_action(next_act, next_act_param, action_dim)
                    agent.step(
                        state,
                        (act, all_action_parameters),
                        reward,
                        next_state,
                        (next_act, next_all_action_parameters),
                        terminal,
                        steps,
                    )

                    act, act_param, all_action_parameters = (
                        next_act,
                        next_act_param,
                        next_all_action_parameters,
                    )
                    action = next_action
                    state = next_state
                    episode_reward += reward

                    if terminal:
                        break

                label.append(int(terminal))
                agent.end_episode()
                returns.append(episode_reward)
                total_reward += episode_reward

                if episode % self.cfg.general.evaluation_interval == 0:

                    agent_val.copy_models(agent.actor, agent.actor_param)

                    if previous_reward > np.array(returns[-100:]).mean() + 0.3:
                        count += 1

                    previous_reward = np.array(returns[-100:]).mean()
                    if count >= 5 and episode > 30000:
                        agent.learning_rate_actor = max(
                            agent.learning_rate_actor / 2,
                            self.cfg.learning_rate_actor_min,
                        )
                        agent.learning_rate_actor_param = max(
                            agent.learning_rate_actor_param / 2,
                            self.cfg.learning_rate_actor_param_min,
                        )
                        count = 0

            print(
                f"Counterfactual: {self._mlmodel.predict(x_best.reshape(1, -1))}\nFactual: {self._mlmodel.predict(x.reshape(1, -1))}\nDistance: {np.linalg.norm((x-x_best))}"
            )
            cf_list.append((x, x_best))


def pad_action(act, act_param, action_dim):
    """
    Pads the action with zeros based on the specified action dimension.

    Args:
        act (int): The action index.
        act_param (float): The parameter associated with the action.
        action_dim (int): The total number of possible actions.

    Returns:
        tuple: A tuple containing the action and a list of parameters for each action.
    """
    # Create an array of zeros for each action parameter
    params = np.zeros((action_dim, 1), dtype=np.float32)

    # Set the parameter for the specified action
    params[act] = act_param

    return act, params
