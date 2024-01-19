from carla.data.catalog import OnlineCatalog
from carla import MLModelCatalog
from carla.recourse_methods import Dice

if __name__ == "__main__":

    # 1. Load data set from the CsvCatalog
 
    dataset = OnlineCatalog(data_name="adult")

    # 2. Load pre-trained black-box model from the MLModelCatalog
    model = MLModelCatalog(
        dataset, load_online=False, model_type="ann", backend="pytorch"
    )
    model.train(0.001, 10, 64, force_train=True)

    # 3. Load recourse model with model specific hyperparameters
    hyperparameters = {}
    gs = Dice(model, hyperparameters)

    # 4. Generate counterfactual examples
    factuals = dataset.df.sample(10)
    counterfactuals = gs.get_counterfactuals(factuals)

    print(counterfactuals)
