from tensorflow import Graph, Session    
from carla.data.catalog import OnlineCatalog
from carla.models.catalog.catalog import MLModelCatalog
from carla.models.negative_instances.predict import predict_negative_instances
from carla.recourse_methods.catalog.cem.model import CEM


if __name__ == "__main__":
    data_name = "adult"
    data = OnlineCatalog(data_name=data_name)

    hyperparams_cem = {
        "beta": 0.0,
        "gamma": 6.0,
        "data_name": data_name,
    }

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data,
                model_type="ann",
                encoding_method="Binary",
                backend="tensorflow",
            )

            factuals = predict_negative_instances(model_ann, data.df)
            test_factuals = factuals.iloc[:5]

            recourse = CEM(
                sess=ann_sess,
                mlmodel=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

            print(counterfactuals_df)