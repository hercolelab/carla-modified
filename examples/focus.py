from carla.data.catalog import OnlineCatalog
from carla.models.catalog.catalog import MLModelCatalog
from carla.models.negative_instances.predict import predict_negative_instances
from carla.recourse_methods.catalog.focus.model import FOCUS


if __name__ == "__main__":

    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, "forest", "xgboost", load_online=False)
    model.train(max_depth=2, n_estimators=5)

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    focus = FOCUS(model)
    cfs = focus.get_counterfactuals(test_factual)

    print(cfs)