import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.linear_layer1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(hidden_features, output_features)
        print(self.linear_layer1)
        print(self.relu)
        print(self.linear_layer2)

    def forward(self, x):
        x = self.linear_layer1(x)
        print(f"after linear_layer1 shape: {x.shape}")
        x = self.relu(x)
        print(f"after relu shape: {x.shape}")
        x = self.linear_layer2(x)
        print(f"after linear_layer2 shape: {x.shape}")
        return x

if __name__ == "__main__":
    in_size = 784
    hidden_units = 128
    out_size = 10

    mlp_model = SimpleMLP(input_features=in_size, hidden_features=hidden_units, output_features=out_size)
    print("*"*10)

    dummy_mlp_input = torch.randn(32, in_size)
    print(f"dummy_mlp_input shape: {dummy_mlp_input.shape}")

    mlp_output = mlp_model(dummy_mlp_input)
    print(f"mlp_output shape: {mlp_output.shape}")
    print("*"*10)

    for name, param in mlp_model.named_parameters():
        if param.requires_grad:
            print(f"name: {name}, shape: {param.shape}")

    print("*"*10)
    sigmoid_activation = nn.Sigmoid()
    input_tensor = torch.tensor([-1.0, 2.0, 7.0])
    output_tensor = sigmoid_activation(input_tensor)

    print(f"input: {input_tensor}")
    print(f"sigmoid output: {output_tensor}")