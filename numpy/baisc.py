import numpy as np
import matplotlib.pyplot as plt

def np_random_seed():
    np.random.seed(0)
    print(np.random.rand(3))
    print(np.random.rand(3))

    np.random.seed(0)
    print(np.random.rand(3))
    print(np.random.rand(3))

def dot_product():
    from datetime import datetime

    a = np.random.randn(100)
    b = np.random.randn(100)
    T = 100000

    ## python code use for loop
    def slow_dot_product(a,b):
        result = 0
        for i,j in zip(a,b):
            result += i*j
        return result

    dt0 = datetime.now()
    for t in range(T): 
        slow_dot_product(a,b)
    dt1 = datetime.now() - dt0
    print(dt1.total_seconds())

    ## numpy code
    ## use vectorization
    dt0 = datetime.now()
    for t in range(T):
        np.dot(a,b)
    dt2 = datetime.now() - dt0
    print(dt2.total_seconds())

    print("dt1/dt2: ", dt1.total_seconds() / dt2.total_seconds()) #~40

def np_matrix():
    L = [ [1, 2], [3, 4]]
    M = np.array([ [1, 2], [3, 4]])
    print(type(M))

    print(L[0][0])
    print(M[0][0])
    # support tuple
    print(M[(0, 0)])
    
    print(np.zeros(4))
    print(np.zeros((5, 5)))
    
    print(np.ones(4))
    print(np.ones((3, 5)))

    # start, end, num => equal distance number array
    print(np.linspace(0, 10, 10))

    # 4*4 unit matrix
    print(np.eye(4))

    # [0, 1) random number
    print(np.random.rand(1)) 
    # 3*3 [0, 1) matrix
    print(np.random.rand(3, 3))
    # normal distribution random number
    print(np.random.randn(4))
    # 3*3 normal distribution matrix
    print(np.random.randn(3, 3))

    # 10 [1, 100) random number
    print(np.random.randint(1, 100, 10))
    
def np_matrix_operations():
    M = np.array([ [1,2], [3,4] ])
    print(M)
    transposM = M.T
    print(transposM)

def matplotlib_plot():
    x = np.linspace(0,10,100) #from 0 - 10, 100 points
    y = np.sin(x)
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Some function of Time")
    plt.title("My Chart")
    plt.show()

def matplotlib_scatter():
    x = np.linspace(0,10,100) #from 0 - 10, 100 points
    y = np.sin(x)
    plt.scatter(x,y)
    plt.xlabel("Time")
    plt.ylabel("Some function of Time")
    plt.title("My Chart")
    plt.show()

def matplotlib_hist():
    R = np.random.randn(10)
    plt.hist(R)
    plt.show()

if __name__ == "__main__":
    matplotlib_hist()
