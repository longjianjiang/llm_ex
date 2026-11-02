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

def tensor_dim():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = t1 + 2
    print(t2)

    t_sum_dim_0 = torch.sum(t2, dim=0)
    print(t_sum_dim_0)
    t_sum_dim_1 = torch.sum(t2, dim=1)
    print(t_sum_dim_1)

def tensor_elements():
    x_1d = torch.tensor([10, 11, 12, 13, 14])
    print(x_1d)

    first_element = x_1d[0] # type is tensor
    print(f"firt element (x_1d[0]): {first_element}, type: {type(first_element)}")

    last_element = x_1d[-1]
    print(f"last element (x_1d[-1]): {last_element}")

    x_1d[1] = 110
    print(x_1d)

    print("*" * 10)
    x_2d = torch.tensor([
        [1, 2, 3],
        [3, 4, 5],
        [7, 8, 9]
    ])
    element_0_1 = x_2d[0, 0]
    print(f"element_0_1 : {element_0_1}")

    first_row = x_2d[0]
    print(f"first row : {first_row}")

    first_col = x_2d[:, 0]
    print(f"first col : {first_col}")

    x_2d[0, 0] = 0
    print(x_2d)

def tensor_slice():
    # Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    x_1d = torch.arange(10)
    print(x_1d)

    # slice, start:stop:step
    # start default 0; stop default nsize; step default 1;
    # [start, end)
    slice1 = x_1d[2:5]
    print(slice1)

    x_2d = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]])
    slice2 = x_2d[0:2]
    print(slice2)
    slice3 = x_2d[1:3]
    print(slice3)

    # slice2 get slice [1:3]
    sub_tensor1 = x_2d[0:2, 1:3]
    print(sub_tensor1)

    # slice share memory with x_2d
    # use .clone not share memory
    sub_tensor1[0,0] = -1
    print(x_2d)

    sub_tensor2 = x_2d[:, -2:]
    print(sub_tensor2)

def tensor_mask():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(f"data : {data}, data shape : {data.shape}")

    mask = data > 3
    print(f"mask : {mask}, type : {mask.type}, shape : {mask.shape}")

    # data1 is shape[1]
    data1 = data[mask]
    print(f"data1: {data1}, shape : {data1.shape}")

    data[data <= 3] = 0
    print(f"data : {data}, data shape : {data.shape}")

def tensor_arr_index():
    x = torch.arange(10, 20) # Tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    print(f"Original 1D tensor: {x}")

    indices = torch.tensor([0, 4, 2, 2]) # Note the repeated index 2
    selected = x[indices]
    print(f"\nSelected elements using indices {indices}: {selected}")

    # For 2D tensors
    y = torch.arange(12).reshape(3, 4)
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11]]
    print(f"\nOriginal 2D tensor:\n{y}")

    # Select specific rows
    row_indices = torch.tensor([0, 2])
    selected_rows = y[row_indices] # Selects rows 0 and 2
    print(f"\nSelected rows using indices {row_indices}:\n{selected_rows}")

    # Select specific columns
    col_indices = torch.tensor([1, 3])
    selected_cols = y[:, col_indices] # Selects columns 1 and 3 from all rows
    print(f"\nSelected columns using indices {col_indices}:\n{selected_cols}")

    # Select specific elements using pairs of indices
    row_idx = torch.tensor([0, 1, 2])
    col_idx = torch.tensor([1, 3, 0])
    selected_elements = y[row_idx, col_idx] # Selects (0,1), (1,3), (2,0) -> [1, 7, 8]
    print(f"\nSelected specific elements using (row_idx, col_idx):\n{selected_elements}")

def tensor_view():
    # Create a contiguous tensor
    x = torch.arange(12) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    print(f"Original tensor: {x}")
    print(f"Original shape: {x.shape}")
    print(f"Is contiguous? {x.is_contiguous()}")

    # Reshape using view()
    y = x.view(3, 4)
    print("\nTensor after view(3, 4):")
    print(y)
    print(f"New shape: {y.shape}")
    print(f"Shares storage with x? {y.storage().data_ptr() == x.storage().data_ptr()}") # Check if they share memory
    print(f"Is y contiguous? {y.is_contiguous()}")

    # Using -1 for inference
    w = x.view(2, 2, -1) # Infers the last dimension to be 3 (12 / (2*2) = 3)
    print("\nTensor after view(2, 2, -1):")
    print(w)
    print(f"New shape: {w.shape}")

def tensor_cat_stack_split():
    # Create two tensors
    tensor_a = torch.randn(2, 3)
    tensor_b = torch.randn(2, 3)
    print(f"Tensor A (Shape: {tensor_a.shape}):\n{tensor_a}")
    print(f"Tensor B (Shape: {tensor_b.shape}):\n{tensor_b}\n")

    # Concatenate along dimension 0 (rows)
    # Resulting shape: (2+2, 3) = (4, 3)
    cat_dim0 = torch.cat((tensor_a, tensor_b), dim=0)
    print(f"Concatenated along dim=0 (Shape: {cat_dim0.shape}):\n{cat_dim0}\n")

    # Concatenate along dimension 1 (columns)
    # Tensors must match in other dimensions (dim 0)
    # Resulting shape: (2, 3+3) = (2, 6)
    cat_dim1 = torch.cat((tensor_a, tensor_b), dim=1)
    print(f"Concatenated along dim=1 (Shape: {cat_dim1.shape}):\n{cat_dim1}")

    # Example with 3D tensors
    tensor_c = torch.randn(1, 2, 3)
    tensor_d = torch.randn(1, 2, 3)
    print(f"tensor c : {tensor_c}, tensor d : {tensor_d}")
    # Concatenate along dim 0 (batch dimension)
    # Resulting shape: (1+1, 2, 3) = (2, 2, 3)
    cat_3d_dim0 = torch.cat((tensor_c, tensor_d), dim=0)
    print(f"\nConcatenated 3D along dim=0 {cat_3d_dim0} (Shape: {cat_3d_dim0.shape})")

if __name__ == "__main__":
    tensor_cat_stack_split()