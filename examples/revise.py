from carla.data.catalog import OnlineCatalog
from carla import MLModelCatalog
from carla.models.negative_instances.predict import predict_negative_instances
from carla.recourse_methods import Revise

if __name__ == "__main__":

    # 1. Load data set from the CsvCatalog
 
    dataset = OnlineCatalog(data_name="adult")

    # 2. Load pre-trained black-box model from the MLModelCatalog
    model = MLModelCatalog(dataset, model_type="ann", backend="pytorch")

    factuals = predict_negative_instances(model, dataset.df)
    test_factual = factuals.iloc[:5]

    vae_params = {
        "layers": [sum(model.get_mutable_mask()), 512, 256, 8],
        "epochs": 1,
    }

    hyperparams = {
        "data_name": "adult",
        "vae_params": vae_params,
    }

    revise = Revise(model, dataset, hyperparams)
    df_cfs = revise.get_counterfactuals(test_factual)

    print(df_cfs)
