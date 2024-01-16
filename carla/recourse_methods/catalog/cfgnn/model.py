"""
Code adapted from: CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks
Link ArXiv: https://arxiv.org/abs/2102.03322

"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse

from carla.recourse_methods.api import RecourseMethod

from .gcn_perturb import GCNSyntheticPerturb
from .utils import get_degree_matrix, get_neighbourhood, get_S_values, normalize_adj


class CFExplainer(RecourseMethod):
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(self, model, beta=None, device=None):
        super(CFExplainer, self).__init__(mlmodel=model)
        self.model = model
        self.beta = beta
        self.device = device

    def explain(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float,
        n_momentum: float,
        num_epochs: int,
        verbose: bool = True,
    ):
        r"""Explain a factual instance:

        Args:
                cf_optimizer (str): choose the optimizer between (Adadelta, SDG)
                node_idx (bool): if true shows more infos about the training phase
                new_idx (int)
                lr (float)
                n_momentum (float): only applied with SDG, it is the Nestor Momentum
                num_epochs (int): Epoch numbers
                verbose (bool)

        """
        # Save new and old index
        self.node_idx = torch.tensor(node_idx)
        self.new_idx = new_idx

        # Save the nodes features (just the ones in the subgraph)
        self.x = self.sub_feat

        # Save the sub adjacency matrix and compute the degree matrix
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        # choose the optimizer
        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0

        for epoch in range(num_epochs):
            new_example, loss_total = self.train(epoch)

            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

        if verbose:
            print(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}\n")

        return best_cf_example

    def train(self, epoch: int, verbose: bool = True) -> Tuple[List, float]:
        r"""Train the counterfactual model:

        Args:
                epoch (int): The epoch number
                verbose (bool): if true shows more infos about the training phase

        """

        # Set the cf model in training mode
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.x, self.A_x)
        output_actual, self.P = self.cf_model.forward_prediction(self.x)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # compute the loss function and perform optim step
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output[self.new_idx], self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        if verbose:

            print(
                f"Node idx: {self.node_idx}",
                f"New idx: {self.new_idx}",
                "Epoch: {:04d}".format(epoch + 1),
                "loss: {:.4f}".format(loss_total.item()),
                "pred loss: {:.4f}".format(loss_pred.item()),
                "graph loss: {:.4f}".format(loss_graph_dist.item()),
            )

            print(
                f"Output: {output[self.new_idx].data}\n",
                f"Output nondiff: {output_actual[self.new_idx].data}\n",
                f"orig pred: {self.y_pred_orig}, new pred: {y_pred_new}, new pred nondiff: {y_pred_new_actual}\n",
            )

        cf_stats = []

        # if a cf example has been found then add it to the cf_stats list
        if y_pred_new_actual != self.y_pred_orig:

            cf_stats = [
                self.node_idx.item(),
                self.new_idx.item(),
                cf_adj.detach().numpy(),
                self.sub_adj.detach().numpy(),
                self.y_pred_orig.item(),
                y_pred_new.item(),
                y_pred_new_actual.item(),
                self.sub_labels[self.new_idx].numpy(),
                self.sub_adj.shape[0],
                loss_total.item(),
                loss_pred.item(),
                loss_graph_dist.item(),
            ]

        return (cf_stats, loss_total.item())

    def get_counterfactuals(
        self, factual: Tuple, i: int, cfg: str, verbose: bool = True
    ):
        r"""Train the counterfactual model:

        Args:
                epoch (int): The epoch number
                verbose (bool): if true shows more infos about the training phase

        """
        import json

        assert cfg is not None, "cfg path should be defined"

        f = open(cfg)
        config = json.load(f)
        f.close()

        self.n_hid = config["n_hid"]
        self.dropout = config["dropout"]
        self.num_classes = config["num_classes"]

        adj = torch.Tensor(factual["adj"]).squeeze()  # Does not include self loops
        features = torch.Tensor(factual["feat"]).squeeze()
        labels = torch.tensor(factual["labels"]).squeeze()
        idx_train = torch.tensor(factual["train_idx"])
        idx_test = torch.tensor(factual["test_idx"])
        edge_index = dense_to_sparse(adj)  # Needed for pytorch-geo functions
        norm_adj = normalize_adj(adj)  # According to reparam trick from GCN paper

        output = self.model.predict_proba(features, norm_adj)

        # Argmax
        y_pred_orig = torch.argmax(output, dim=1)

        # Get the subgraph from the original graph induced by the node node_idx
        sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
            node_idx=int(i),
            edge_index=edge_index,
            n_hops=config["n_layers"] + 1,
            features=features,
            labels=labels,
        )

        new_idx = node_dict[int(i)]

        self.sub_adj = sub_adj
        self.sub_feat = sub_feat
        self.sub_labels = sub_labels
        self.y_pred_orig = y_pred_orig[new_idx]

        # Instantiate CF model class, load weights from original model
        # The syntentic model load the weights from the model to explain then freeze them
        # and train the perturbation matrix to change the prediction
        self.cf_model = GCNSyntheticPerturb(
            nfeat=self.sub_feat.shape[1],
            nhid=config["n_hid"],
            nout=config["n_hid"],
            nclass=config["num_classes"],
            adj=self.sub_adj,
            dropout=config["dropout"],
            beta=config["beta"],
        )

        self.cf_model.load_state_dict(self.model.raw_model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

        if verbose:

            # Check the gradient for each parameter
            for name, param in self.model.raw_model.named_parameters():
                print("orig model requires_grad: ", name, param.requires_grad)
            for name, param in self.cf_model.named_parameters():
                print("cf model requires_grad: ", name, param.requires_grad)

            print(f"y_true counts: {np.unique(labels.numpy(), return_counts=True)}")
            print(
                f"y_pred_orig counts: {np.unique(y_pred_orig.numpy(), return_counts=True)}"
            )  # Confirm model is actually doing something

            # Check that original model gives same prediction on full graph and subgraph
            with torch.no_grad():
                print(f"Output original model, full adj: {output[i]}")
                print(
                    f"Output original model, sub adj: {self.model.predict_proba(sub_feat, normalize_adj(sub_adj))[new_idx]}"
                )

        # If cuda is avaialble move the computation on GPU
        if self.device == "cuda":
            self.model.cuda()
            self.cf_model.cuda()
            adj = adj.cuda()
            norm_adj = norm_adj.cuda()
            features = features.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_test = idx_test.cuda()

        # node to explain i, node_dict maps the old node_idx into the new node_idx
        # because of the subgraph
        cf_example = self.explain(
            node_idx=i,
            cf_optimizer=config["optimizer"],
            new_idx=new_idx,
            lr=config["lr"],
            n_momentum=config["n_momentum"],
            num_epochs=config["num_epochs"],
        )

        return cf_example
