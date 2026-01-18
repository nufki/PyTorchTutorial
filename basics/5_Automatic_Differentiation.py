###############################################################################
# Automatic Differentiation with torch.autograd
###############################################################################

# When training neural networks, the most frequently used algorithm is back propagation. In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.
#
# To compute those gradients, PyTorch has a built-in differentiation engine called torch.autograd. It supports automatic computation of gradient for any computational graph.
#
# Consider the simplest one-layer neural network, with input x, parameters w and b, and some loss function.
# It can be defined in PyTorch in the following manner:

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w)+b # does not yet solve for z
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # now it solves for z
# loss = torch.nn.functional.mse_loss(z, y)

print(f"z: {z}")
print(f"w: {w}")

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing Gradients
# To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function
# with respect to parameters, namely, we need to derive the loss with respect to w and b.
# under some fixed values of x and y. To compute those derivatives, we call loss.backward(),
# and then retrieve the values from w.grad and b.grad:

loss.backward()
print(w.grad)
print(b.grad)

# verify the result, true means it successfully solve the equation
z2 = w.sum(dim=0) + b
print(torch.allclose(z, z2))
print(z, z2)


# Disabling Gradient Tracking
# By default, all tensors with requires_grad=True are tracking their computational history and support gradient
# computation. However, there are some cases when we do not need to do that, for example, when we have trained
# the model and just want to apply it to some input data, i.e. we only want to do forward computations through
# the network. We can stop tracking computations by surrounding our computation code with torch.no_grad() block:
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor:
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# There are reasons you might want to disable gradient tracking:
# To mark some parameters in your neural network as frozen parameters.
#
# To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.

