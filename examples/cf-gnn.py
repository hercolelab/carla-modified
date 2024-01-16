import pickle

from carla.models.catalog.GNN_TORCH.model_gnn import TreeGridModel
from carla.recourse_methods.catalog.cfgnn.model import CFExplainer

if __name__ == "__main__":

    # Load the dataset
    with open("carla/data/syn4.pickle", "rb") as f:
        factual = pickle.load(f)

        # Load the GNN model to explain
    # IMPORTANT: the model to explain must be implemented using carla MLModel API
    model = TreeGridModel()

    # Load the explainer
    explainer = CFExplainer(model=model, device="cpu")

    # Explain all the sample in the testset
    # Important: you should remeber to build and load a proper configuration
    for i in range(len(factual["test_idx"])):

        example_id = factual["test_idx"][i]

        # Important cf-gnn is used to perform cf-explaination on node classification.
        # The dataset is a single large graph. The node to explain is passed as an integer (example_id).
        # During the preprocessing the example_id is used to extract the subgraph
        explainer.get_counterfactuals(
            factual,
            example_id,
            cfg="carla/recourse_methods/catalog/cfgnn/cfgs/syn4.json",
        )
