import pandas as pd
from xgboost import XGBClassifier

from carla.recourse_methods.catalog.relax.model import Relax

if __name__ == "__main__":

    # Important: to modify the parameters you must go to relax/conf and modify the .yaml files

    # Declare a model
    model = XGBClassifier(max_depth=5, n_estimators=600)

    # Import the dataset
    factuals = pd.read_csv("carla/recourse_methods/catalog/relax/dataset/Covid.csv")
    x_train, y_train = (
        factuals.iloc[:110, :-1].to_numpy(),
        factuals.iloc[:110, -1].to_numpy(),
    )

    # Fit the model
    model.fit(x_train, y_train)

    # Declatre the explainer
    recourse = Relax(mlmodel=model)

    # Get the counterfactual
    recourse.get_counterfactuals(factuals=factuals.iloc[111:])
