import torch
import torch.optim as optim

# learning hours
x = [1.0, 2.0]#, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# scores
y = [45.0, 50.0]#, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]

t_x = torch.tensor(x)
t_y = torch.tensor(y)
t_xn = t_x*0.1

def model(t_x, w, b):
    return w * t_x + b

def loss_fn(t_y, t_p):
    diff = (t_p - t_y) ** 2
    return diff.mean()

def dloss_fn(t_y, t_p):
    return 2 * (t_p - t_y)

def dmodel_dw(t_x, w, b):
    return t_x

def dmodel_db(t_x, w, b):
    return 1.0

def grad_fn(t_x, w, b, t_y, t_p):
    dw = dloss_fn(t_y, t_p) * dmodel_dw(t_x, w, b)
    db = dloss_fn(t_y, t_p) * dmodel_db(t_x, w, b)
    return torch.tensor([dw.mean(), db.mean()])

def train(learning_rate, w, b, x, y):
    t_p = model(x, w, b)
    loss = loss_fn(y, t_p)
    print(f"w = {w}, b = {b}")
    print(f"loss : {float(loss)}")
    gradient_w, gradient_b = grad_fn(t_x, w, b, t_y, t_p)
    print(f"gradient_w : {gradient_w}, gradient_b : {gradient_b}")
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b
    print(f"w = {w}, b = {b}")

    t_p = model(x, w, b)
    loss = loss_fn(y, t_p)
    print(f"loss : {float(loss)}")

    return (w, b)

learning_rate = 0.01
initial_w = 0
initial_b = 0
train(learning_rate, initial_w, initial_b, t_x, t_y)