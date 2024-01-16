from copy import deepcopy

import numpy as np
import torch as torch
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier


class Environment:
    def __init__(self, observation_dim, action_dim, model, balance_factor, max_step):
        self.act_dim = action_dim
        self.obs_dim = observation_dim
        self.model = model
        self.timestep = 0
        self.gamma = balance_factor
        self.max_step = max_step
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.candidate = []
        self.gap = 0

    def reset(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.x_ori = x
        self.y_ori = y
        self.state = x
        self._action_available = np.arange(self.act_dim)
        return self.state

    def reset_local(self, x, y, i):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            i (_type_): _description_

        Returns:
            _type_: _description_
        """

        self.x_ori = x
        self.y_ori = y
        self.state = x
        self._action_available = self.select_feature_local(i)
        prob = np.squeeze(self.model.predict_proba(self.state.reshape(1, -1).numpy()))
        self.gap = np.abs(prob[0] - prob[1])

        return self.state

    def build_feature_local(self, X, Y, n):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            n (_type_): _description_
        """

        nbrs = NearestNeighbors(n_neighbors=n, algorithm="kd_tree").fit(X)
        distances, indices = nbrs.kneighbors(X)
        X = torch.FloatTensor(deepcopy(X))
        label_matrix = Y[indices]

        candidate = []

        num_feat = 16

        for i in range(X.shape[0]):
            clf = DecisionTreeClassifier(random_state=0, max_depth=10)
            X_loc = X[indices[i], :]
            Y_loc = label_matrix[indices[i], :]
            clf.fit(X_loc, Y_loc)
            importance = clf.feature_importances_
            importance = np.argsort(importance)[::-1]
            candidate.append(importance[:num_feat])
        self.candidate = candidate

    def build_meta_local(self, X):
        """_summary_

        Args:
            X (_type_): _description_
        """

        common = np.array([1, 2, 9, 10, 11, 12, 13, 15])

        self.candidate = [common for _ in range(X.shape[0])]

    def select_feature_local(self, i):
        """_summary_

        Args:
            i (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.candidate[i]

    def step(self, action):
        """
        Executes a step in the environment based on the given action.

        This method increments the internal timestep, processes the specified action,
        and updates the state of the environment accordingly. It calculates the new
        state based on the action and the previous state, computes the reward using
        the reward shaping function, and determines if the current state is terminal.

        Parameters:
        action (tuple): A tuple containing the action index and its corresponding value.

        Returns:
        tuple: A tuple containing the new state and timestep, the computed reward,
            and a boolean indicating whether the state is terminal.
        """
        action_bound = 5
        self.timestep += 1
        i = action[0]
        x = min(action[1][i][0], action_bound)

        if i in self._action_available:

            t = np.where(self._action_available == i)[0][0]
            self._action_available = np.delete(self._action_available, t)
            state_last = self.state[i]
            self.state[i] += x
            diff = self.gamma * (
                np.abs(state_last - self.x_ori[i])
                - np.abs(self.state[i] - self.x_ori[i])
            )
            label, change = self.reward_shape()
            reward = label * 18 + diff

            return ((self.state, self.timestep), reward, change)

    def reward(self):
        """
        Compute the reward function
        """
        out = self.model.predict(self.state)

        return (0, False) if out == self.y_ori else (1, True)

    def reward_shape(self):
        """
        Computes and returns the reward based on the current state and the model's predictions.

        This method adjusts the current state to handle any '-inf' values, making them suitable for model prediction.
        It then predicts the outcome and calculates the reward based on the change in prediction probability gap.
        The method also updates the internal gap value for future calculations and determines whether the current
        state leads to a terminal condition.

        Returns:
        tuple: A tuple containing the calculated reward and a boolean indicating whether the current state is terminal.
        """

        x = self.state.reshape(1, -1).numpy()
        out = self.model.predict(x)

        if out != self.y_ori:
            return 1, True

        else:
            prob = np.squeeze(self.model.predict_proba(x))
            gap_t = np.abs(prob[0] - prob[1])
            r = self.gap - gap_t
            self.gap = gap_t

            return r, False

    @property
    def actions_available(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return self._action_available
