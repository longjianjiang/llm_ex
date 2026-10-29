import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# learning hours
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# scores
y = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

# fig, ax = plt.subplots()
# ax.scatter(X, y, color='#1c7ed6', label='student data') # 使用蓝色
# ax.set_xlabel("learning hours")
# ax.set_ylabel("scores")
# ax.set_title("scores vs learning hours")
# ax.legend()
# ax.grid(True)
# plt.show()

def calculate_cost(X, y, m, b):
    predictions = m * X + b

    return np.mean((predictions - y) ** 2)

# (mx + b - y)^2
# 2(mx + b - y) * x
# 2(mx + b - y) * 1
def compute_gradient(X, y, m, b):
    predictions = m * X + b
    error = predictions - y
    
    dl_dm = 2 * np.mean(X * error)
    dl_db = 2 * np.mean(error)

    return dl_dm, dl_db

def gradient_descent(X, y, initial_m, initial_b, learnint_rate, iterations):
    m = initial_m
    b = initial_b
    nsize = len(X)
    cost_history = []

    for i in range(iterations):
        predictions = m * X + b
        error = y - predictions

        # gradient_m = (-2/nsize) * np.sum(X * error)
        # gradient_b = (-2/nsize) * np.sum(error)
        gradient_m, gradient_b = compute_gradient(X, y, m, b)

        m = m - learnint_rate * gradient_m
        b = b - learnint_rate * gradient_b

        cost = calculate_cost(X, y, m, b)
        cost_history.append(cost)

        if (i+1) % 100 == 0:
             print(f"iterate {i+1}/{iterations}, cost (m={m}, b={b}): {cost:.4f}")
        
    return m, b, cost_history

initial_m = 0
initial_b = 0
initial_cost = calculate_cost(X, y, initial_m, initial_b)
print(f"initial cost (m=0, b=0): {initial_cost}")

learning_rate = 0.01
iterations = 1000
final_m, final_b, cost_history = gradient_descent(X, y, initial_m, initial_b, learning_rate, iterations)

print(f"\ntrain done。")
print(f"final (m): {final_m:.4f}")
print(f"final (b): {final_b:.4f}")
print(f"final (MSE): {cost_history[-1]:.4f}")