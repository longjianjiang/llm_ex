import torch

def tensor_create_from_data():
    data_list = [1, 2]
    tensor_from_list = torch.tensor(data_list)

    print(tensor_from_list)
    print(f"data type: {tensor_from_list.dtype}")
    print(f"shape: {tensor_from_list.shape}")

def tensor_create():
    shape = (2, 3)
    
    zeros_tensor = torch.zeros(shape)
    ones_tensor = torch.ones(shape)
    empty_tensor = torch.empty(shape)

    print(zeros_tensor)
    print(f"zeros shape : {zeros_tensor.shape}")

    print(ones_tensor)
    print(f"ones shape : {ones_tensor.shape}")

    print(empty_tensor)
    print(f"empty shape : {empty_tensor.shape}")

    ones_int_tensor = torch.ones(shape, dtype=torch.int32)
    print(ones_int_tensor)
    print(f"ones_int shape : {ones_int_tensor.shape}")

    rand_tensor = torch.rand(shape)
    randn_tensor = torch.randn(shape)

    print(rand_tensor)
    print(randn_tensor)

def tensor_multiply():
    x = [1.0, 2.0]
    t_x = torch.tensor(x)

    tmp1 = t_x * 2 + 1
    # tensor([3, 5])
    print(tmp1)

    y = [10.0, 20.0]
    t_y = torch.tensor(y)

    diff = (t_x - t_y)
    print(diff)
    tmp2 = diff ** 2
    print(tmp2)
    tmp3 = diff * 2
    print(tmp3)
    tmp4 = tmp2.mean()
    print(tmp4)

if __name__ == "__main__":
    tensor_multiply()