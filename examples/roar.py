from carla.data.catalog import OnlineCatalog
from carla import MLModelCatalog
from carla.models.negative_instances.predict import predict_negative_instances
from carla.recourse_methods import Roar
import pandas as pd

if __name__ == "__main__":

    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, "ann", backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:10]

    hyperparams = {"loss_type": "MSE", "y_target": [1]}
    df_cfs = Roar(model, hyperparams).get_counterfactuals(test_factual)
    print(df_cfs)


