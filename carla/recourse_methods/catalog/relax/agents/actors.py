import torch
import torch.nn as nn
import torch.nn.functional as F


class QActor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        action_parameter_size,
        hidden_layers=(100,),
        action_input_layer=0,
        output_layer_init_std=None,
        activation="relu",
        **kwargs
    ):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            print("inputsize", inputSize, "hidden_layers[0]", hidden_layers)
            self.layers.append(
                nn.Linear(inputSize, hidden_layers[0], dtype=torch.float64)
            )
            for i in range(1, nh):
                self.layers.append(
                    nn.Linear(
                        hidden_layers[i - 1], hidden_layers[i], dtype=torch.float64
                    )
                )
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(
            nn.Linear(lastHiddenLayerSize, self.action_size, dtype=torch.float64)
        )

        # initialise layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0.0, std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        """ """

        negative_slope = 0.01
        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        x = x.to(torch.float64)

        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        action_parameter_size,
        hidden_layers,
        squashing_function=False,
        output_layer_init_std=None,
        init_type="kaiming",
        activation="relu",
        init_std=None,
    ):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        print("state_size", state_size)
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(
                nn.Linear(inputSize, hidden_layers[0], dtype=torch.float64)
            )
            for i in range(1, nh):
                self.layers.append(
                    nn.Linear(
                        hidden_layers[i - 1], hidden_layers[i], dtype=torch.float64
                    )
                )
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(
            lastHiddenLayerSize, self.action_parameter_size, dtype=torch.float64
        )
        self.action_parameters_passthrough_layer = nn.Linear(
            self.state_size, self.action_parameter_size, dtype=torch.float64
        )

        # initialise layer weights
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type " + str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(
                self.action_parameters_output_layer.weight, std=output_layer_init_std
            )
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, x):

        x_copy = x.to(torch.float64)
        negative_slope = 0.01
        x = x.to(torch.float64)
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(x_copy)

        return action_params
