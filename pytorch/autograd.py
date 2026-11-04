import torch

def tensor_autograd():
    # loss = (w * x + b) ** 2

    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(3.0)

    tmp = w * x + b
    loss = tmp ** 2
    print(tmp.grad_fn)
    print(loss.grad_fn)

    loss.backward(retain_graph=True)
    # can repeat
    loss.backward()
    print(w.grad)
    print(b.grad)

    w.grad.zero_()
    b.grad.zero_()
    
    print(w.grad)
    print(b.grad)

def tensor_detach():
    b = torch.tensor(1.0, requires_grad=True)
    # c requires_grad is False
    c = b.detach()
    print(f"c.requires_grad: {c.requires_grad}")
    
    d = c + 1
    print(f"d.requires_grad: {d.requires_grad}")

    e = b + 1
    print(f"e.requires_grad: {e.requires_grad}")
    print(e.grad_fn)

if __name__ == "__main__":
    tensor_autograd()