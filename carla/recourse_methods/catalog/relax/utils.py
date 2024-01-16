import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from texttable import Texttable
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = len(X)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        - An integer count of the number of samples.
        """
        return len(self.X)

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        Parameters:
        - index: The index of the sample to retrieve.

        Returns:
        - A tuple (X, y) where X is the feature data and y is the label for the specified sample.
        """
        X = self.X[index]
        y = self.y[index]
        return X, y

    def sample(self, ratio):
        """
        Randomly sample a subset of the dataset based on a specified ratio.

        Parameters:
        - ratio: A float representing the fraction of the dataset to sample (0 < ratio <= 1).

        Returns:
        - A new Data object containing the sampled subset of data.
        """

        idx = list(range(self.N))
        np.random.shuffle(idx)
        idx = idx[: int(self.N * ratio)]
        X = self.X[idx]
        y = self.y[idx]
        return Data(X, y)

    def limit(self, num: int) -> None:
        """
        Limit the dataset to a specified number of samples. If the specified number is less than
        the total number of samples, a stratified shuffle split is performed to maintain the
        distribution of labels in the reduced dataset.

        Parameters:
        - num: An integer specifying the maximum number of samples to retain in the dataset.

        Prints the size of the dataset before and after the split.
        """

        print("before split", len(self.X))
        idx = np.random.choice(
            list(range(len(self.X))), min(num, len(self.X)), replace=False
        )
        split = 1.0 * min(num, len(self.X)) / len(self.X)
        if split != 1.0:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=(1 - split), random_state=77
            )
            for train_idx, test_idx in sss.split(self.X, self.y):
                train_x, test_x = self.X[train_idx], self.X[test_idx]
                train_y, test_y = self.y[train_idx], self.y[test_idx]
            self.X = train_x
            self.y = train_y
        print("after split", len(self.X))

    def getX(self):
        """
        Get the feature data of the dataset.

        Returns:
        - The feature data X as a numpy array or similar data structure.
        """

        return self.X

    def gety(self):
        """
        Get the label data of the dataset.

        Returns:
        - The label data y as a numpy array or similar data structure.
        """

        return self.y


def print_performance(val_acc, val_auc, test_acc, test_auc):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(
        [
            "t",  # text
            "f",  # float (decimal)
            "f",  # float (decimal)
        ]
    )  # automatic
    table.set_cols_align(["l", "c", "c"])
    table.add_rows(
        [
            ["Dataset", "Accuracy", "F1"],
            ["Validation", val_acc, val_auc],
            ["Test", test_acc, test_auc],
        ]
    )
    print("\n", table.draw())


def print_results(avgfeat, fidelity):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(
        [
            "t",  # text
            "f",  # float (decimal)
            "f",  # float (decimal)
        ]
    )  # automatic
    table.set_cols_align(["l", "c", "c"])
    table.add_rows(
        [["Dataset", "#avgFeatChanged", "Fidelity"], ["Test", avgfeat, fidelity]]
    )

    print("\n", table.draw())


def check_if_col_int(col):
    col = col.flatten()
    for i in col:
        if i % 1 != 0:
            return False
    return True


def get_constraints(train_x):
    bound_min = np.amin(train_x, axis=0)
    bound_max = np.amax(train_x, axis=0)
    type_int = []
    for i in range(train_x.shape[1]):
        type_int.append(check_if_col_int(train_x[:, i]))
    return bound_min, bound_max, type_int


def read_data(file, seed=77, scaler=False, test_split=0.1, validation_split=0.1):
    remove_cols = ["id", "ID", "Id", "Unnamed: 32"]
    data = pd.read_csv("./src/dataset/{}".format(file), sep=",")
    scaler = StandardScaler() if scaler else None
    le = LabelEncoder()

    for col in remove_cols:
        try:
            data = data.drop([col], axis=1)
        except:
            pass

    data = data.fillna(method="backfill")
    data = shuffle(data, random_state=seed)
    feat_df = data.drop(["class"], axis=1)
    features = feat_df.columns.values
    x = feat_df.values
    y = data[["class"]].values.flatten()
    y = le.fit_transform(y)

    # split training and testing set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    for train_idx, test_idx in sss.split(x, y):
        train_x, test_x = x[train_idx], x[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_split, random_state=seed
    )
    for train_idx, test_idx in sss.split(train_x, train_y):
        train_x, val_x = train_x[train_idx], train_x[test_idx]
        train_y, val_y = train_y[train_idx], train_y[test_idx]

    if scaler:
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        val_x = scaler.transform(val_x)

    train_data = Data(train_x, train_y)
    test_data = Data(test_x, test_y)
    val_data = Data(val_x, val_y)

    return scaler, le, x, y, features, train_data, val_data, test_data
