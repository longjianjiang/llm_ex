import numpy as np

def testCausal():
    T = 5
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    print(mask)

def dotProduct():
    v1 = np.array([1, 0.9])
    v2 = np.array([0.9, 0.7])
    v3 = np.array([0.2, 0.1])

    # a bigger dot product means v1 and v2 are more similar
    print(np.dot(v1, v2))
    print(np.dot(v1, v3))

# \text{Masked Attention}(Q, K, V, M) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M \right) V
def self_attention(x, W_q, W_k, W_v, W_o):
    batch, token, embeddingDims = x.shape
    dims_k = W_q.shape[1]

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    # scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dims_k)

    # causal
    mask = np.triu(np.ones((token, token), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    # softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores - scores_max)
    attn = exp / exp.sum(axis=-1, keepdims=True)

    out = attn @ V
    return out

if __name__ == "__main__":
    self_attention()
