from xgboost import XGBClassifier

from carla import CsvCatalog, MLModelCatalog, OnlineCatalog
from carla.recourse_methods import GrowingSpheres

if __name__ == "__main__":

    target = "income"
    continuous = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "hours-per-week",
        "capital-loss",
    ]
    categorical = [
        "marital-status",
        "native-country",
        "occupation",
        "race",
        "relationship",
        "sex",
        "workclass",
    ]
    immutable = ["age", "sex"]

    # 1. Load data set from the OnlineCatalog
    dataset = CsvCatalog(
        "examples/data/adult.csv", categorical, continuous, immutable, target
    )

    # 2. Load pre-trained black-box model from the MLModelCatalog
    model = MLModelCatalog(
        dataset, load_online=False, model_type="ann", backend="pytorch"
    )
    model.train(0.001, 10, 64)

    # 3. Load recourse model with model specific hyperparameters
    hyperparameters = {}
    gs = GrowingSpheres(model, hyperparameters)

    # 4. Generate counterfactual examples
    factuals = dataset.df.sample(10)
    counterfactuals = gs.get_counterfactuals(factuals)

    print(counterfactuals)
