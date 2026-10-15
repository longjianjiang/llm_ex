import torch
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.W_q = nn.Linear(in_features=embedding_dim, out_features=output_dim, bias=False)
    
    def forward(self, x):
        x = self.W_q(x) 
        return x

if __name__ == "__main__":
    net = LinearNet(embedding_dim=10, output_dim=5)
    x = torch.randn(1, 10)
    print(x)
    print(x.shape)
    print("*"*100)
    print(net(x))
    print(net(x).shape)