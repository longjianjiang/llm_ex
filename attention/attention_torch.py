import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EmbeddingDims = 256 

class SelfAttention(nn.Module):
    def __init__(self, d_model=EmbeddingDims, d_k=128):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_k, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_k, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def forward(self, x):
        # x: (T, d_model)
        q = self.W_q(x)  
        k = self.W_k(x)  
        v = self.W_v(x) 

        sims = q @ k.transpose(-2, -1)
        print(sims.shape)

        scaled_sims = sims / math.sqrt(q.size(-1))

        attn = F.softmax(scaled_sims, dim=-1)
        print(attn.shape)

        out = attn @ v
        return out

def main():
    torch.manual_seed(0)

    T = 8
    x = torch.randn(T, EmbeddingDims) 
    print("input x:", x.shape)

    head = SelfAttention()
    out = head(x)
    print("output out:", out.shape)

if __name__ == "__main__":
    main()
