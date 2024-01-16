# Data Processing
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = len(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def sample(self, ratio):
        idx = list(range(self.N))
        np.random.shuffle(idx)
        idx = idx[: int(self.N * ratio)]
        X = self.X[idx]
        y = self.y[idx]
        return Data(X, y)

    def limit(self, num):
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
        return self.X

    def gety(self):
        return self.y


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
    # scaler = MinMaxScaler() if scaler else None
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


class Dataset:
    def __init__(self, cfg, model) -> None:

        scaler, le, _, _, features, train_data, val_data, test_data = read_data(
            cfg.general.csv, cfg.agent.seed, scaler=cfg.general.pre_scaler
        )

        num_action = train_data.getX().shape[1]
        bound_min, bound_max, bound_type = get_constraints(train_data.getX())

        self.observation_dim = train_data.getX().shape[1]
        self.action_dim = train_data.getX().shape[1]
        self.action_high = bound_max
        self.action_low = bound_min

        self.X_tr = train_data.getX()
        self.X_val = val_data.getX()
        self.X_test = test_data.getX()

        self.tr_size = self.X_tr.shape[0]
        self.val_size = self.X_val.shape[0]
        self.test_size = self.X_test.shape[0]

        self.get_y_from_model(model)

        self.X_tr = np.vstack((self.X_tr, np.vstack((self.X_val, self.X_test))))
        self.Y_tr = np.concatenate(
            (self.y_tr, np.concatenate((self.y_val, self.y_test)))
        )

        self.tr_size = self.X_tr.shape[0]

        self.X1 = self.X_tr[self.Y_tr == 1]
        self.Y1 = self.Y_tr[self.Y_tr == 1]

    def get_y_from_model(self, model):

        self.y_tr = model.predict(self.X_tr)
        self.y_val = model.predict(self.X_val)
        self.y_test = model.predict(self.X_test)

    @property
    def x(self):
        return self.X1

    @property
    def y(self):
        return self.Y1
