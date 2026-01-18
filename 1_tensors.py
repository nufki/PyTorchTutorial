###############################################################################
# Tensors
###############################################################################

import torch

# Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
# Initializing a Tensor - Tensors can be created directly from data. The data type is automatically inferred.
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(x_data)



x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


shape = (2,3,) # 2x3 matrice
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Random Tensor:
# tensor([[0.2534, 0.8156, 0.1311],
#         [0.7459, 0.8381, 0.1763]])
#
# Ones Tensor:
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
#
# Zeros Tensor:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])



tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor Attributes - Tensor attributes describe their shape, datatype, and the device on which they are stored.
# Shape of tensor: torch.Size([3, 4])
# Datatype of tensor: torch.float32
# Device tensor is stored on: cpu


# Operations on Tensors
# Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described here.
# Each of these operations can be run on the CPU and Accelerator such as CUDA, MPS, MTIA, or XPU. If you’re using Colab, allocate an accelerator by going to Runtime > Change runtime type > GPU.
# By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using .to method (after checking for accelerator availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!


# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# First row: tensor([1., 1., 1., 1.])
# First column: tensor([1., 1., 1., 1.])
# Last column: tensor([1., 1., 1., 1.])
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])


# oining tensors You can use torch.cat to concatenate a sequence of tensors along a given dimension.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

#Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

tensor2 = torch.ones(4, 4)

# both are possible...
y1 = tensor2 @ tensor2.T
y2 = tensor2.matmul(tensor2.T)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor2 * tensor2
z2 = tensor2.mul(tensor2)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# calculate the sum of all elements
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))