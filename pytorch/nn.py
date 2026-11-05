import torch
import torch.nn as nn

class CustomModuleWithParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_weight = nn.Parameter(torch.randn(5, 2))
        self.my_info = torch.tensor([1.0, 2.0])

    def forward(self, x):
        return torch.matmul(x, self.my_weight)

module = CustomModuleWithParameter()

for name, param in module.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

class LinearNet(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=input_features, out_features=output_features)
    
    def forward(self, x):
        output = self.linear_layer(x) 
        return output

if __name__ == "__main__":
    in_dim = 10
    out_dim = 1

    model = LinearNet(input_features=in_dim, output_features=out_dim)

    dummy_input = torch.randn(5, in_dim)
    print(f"dummy_input shape: {dummy_input.shape}")
    print(dummy_input)

    output = model(dummy_input)
    print(f"output shape: {output.shape}")
    print(output)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"name: {name}, shape: {param.shape}")
            print(param)