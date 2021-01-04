# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Load dependencies:

# %%
import numpy as np
import tensorflow as tf

# %% [markdown]
# Implementation of Jennrich's algorithm(stable):

# %%
def Jennrich(M1: tf.Tensor, M2: tf.Tensor, r: int):
    # Inputs:
    # M1, M2: d-by-d matrices of the same rank = r
    # Outputs:
    # the eigenvectors of M1*M2^(-1)
    S, U, V = tf.linalg.svd(M1)
    W = U[:, 0 : r]
    M1_whitened = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(W), M1), W)
    M2_whitened = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(W), M2), W)
    M = tf.linalg.matmul(M1_whitened, tf.linalg.inv(M2_whitened))
    e, P = tf.linalg.eig(M)
    return tf.linalg.matmul(W, tf.math.real(P))


# %%
def matSys(A_r, Tx, x):
    # Solve a matrix valued linear system A_r * Xi * diag(A_r.T * x) * A_r.T = Tx
    # Inputs: 
    # A_r: a d-by-r matrix; 
    # Tx: a d-by-d matrix; 
    # x: a d-by-1 vector
    # Outputs:
    # Xi: a d-by-1 vector
    B = tf.transpose(tf.linalg.pinv(A_r))
    xi = tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(B), Tx), B))
    Xi = tf.divide(xi, tf.tensordot(tf.transpose(A_r), x, axes= [[1],[0]]))
    return Xi


# %%
def decompose(T: tf.Tensor, x: tf.Tensor, y: tf.Tensor, rank):
    # Given an input tensor, run Jennrich's algorithm and length recovery
    # Inputs:
    # T: tensor to be decomposed;
    # x,y: vectors used in Jennrich's algorithm; 
    # rank: rank of T.
    # Outputs:
    # A: a matrix with columns as directions of tensor components
    # Xi: a diagonal matrix containing the length of each conponent
    T_x = tf.tensordot(T, x, axes=[[0],[0]])
    T_y = tf.tensordot(T, y, axes=[[0],[0]])
    A = Jennrich(T_x, T_y, r = rank)
    Xi = matSys(A, T_x, x)
    return A, Xi

# %% [markdown]
# Helper function on building the symmetric tensor:

# %%
def constructSymTensor(components: tf.Tensor, weights = None):
    # Construct a 3rd order symmetric tensor with columns of components and muliplicative constants from weights
    # components: an n-by-m matrix with m components in dimension n
    # weights: a m-by-1 vector
    dim = components.shape[0]
    rank = components.shape[1]
    if weights == None:
        weights = tf.ones(shape=[rank,1])
    elif weights.ndim == 1:
        weights = tf.expand_dims(weights, axis = 1)
    T = tf.zeros(shape = [dim,dim,dim])
    for i in range(rank):
        ai = tf.expand_dims(components[:,i], axis = 1)
        T = tf.add(T, tf.einsum('ij,kl->ijk', tf.tensordot(ai, weights[i,0] * ai, axes= [[1],[1]]) , ai))
    return T

# %% [markdown]
# Implementation of accelerated Algorithm 3.3 on Tensorflow:

# %%
class OvercompleteTensorDecomposition(tf.Module):
    # Overcomplete tensor decomposition "layer"
    def __init__(self, dimension, name = None):
        super().__init__(name = name)
        init = tf.random_normal_initializer()
        self.x_magic = tf.Variable(initial_value = init(shape = [dimension], dtype = tf.float32), name = "x_magic")
        self.y_magic = tf.Variable(initial_value = init(shape = [dimension], dtype = tf.float32), name = "y_magic")
        self.x_2nd = tf.constant(value= init(shape = [dimension], dtype= tf.float32), name = "x_2nd")
        self.y_2nd = tf.constant(value= init(shape = [dimension], dtype= tf.float32), name = "y_2nd")
    def __call__(self, T: tf.Tensor, tensor_rank, overcomplete_param):
        # T: a symmetric d-by-d-by-d tensor of rank tensor_rank
        
        # Decompose the first r compoents
        A_r, Xi_r = decompose(T, self.x_magic, self.y_magic, rank = tensor_rank - overcomplete_param)

        # Deflation
        T_first = constructSymTensor(components= A_r, weights= Xi_r)
        R = T - T_first

        # 2nd decomposition
        # x_2nd = tf.random.normal(shape = self.x_magic.shape)
        # y_2nd = tf.random.normal(shape = self.y_magic.shape)
        A_k, Xi_k = decompose(R, self.x_2nd, self.y_2nd, rank = overcomplete_param)
        
        # Reconstruction
        T_second = constructSymTensor(components= A_k, weights= Xi_k)

        return T_first + T_second


# %%
class TensorDecomposition(tf.Module):
    # Non-degenerate tensor decomposition "layer"
    def __init__(self, dimension, name = None):
        init = tf.random_normal_initializer()
        self.x_magic = tf.Variable(initial_value = init(shape = [dimension], dtype = tf.float32), name = "x_magic")
        self.y_magic = tf.Variable(initial_value = init(shape = [dimension], dtype = tf.float32), name = "y_magic")
    def __call__(self, T, tensor_rank):
        # T: a symmetric d-by-d-by-d tensor of rank tensor_rank
        
        # Decompose the tensor
        A_r, Xi_r = decompose(T, self.x_magic, self.y_magic, rank = tensor_rank)

        # Reconstruction
        T_prime = constructSymTensor(components=A_r, weights=Xi_r)
        return T_prime


# %%
def loss_func(T1: tf.Tensor, T2: tf.Tensor):
    return tf.nn.l2_loss(T1 - T2)


# %%
d = 10
model = TensorDecomposition(dimension = d)
print(model.trainable_variables)

# %% [markdown]
# Define the training process

# %%
def train(model, T, rank_T, learning_rate = 1e-5):
    with tf.GradientTape() as t:
        current_loss = loss_func(T, model(T, rank_T))
    dx, dy = t.gradient(current_loss,[model.x_magic, model.y_magic])
    model.x_magic.assign_sub(learning_rate * dx)
    model.y_magic.assign_sub(learning_rate * dy)

# %% [markdown]
# Generate a random tensor(non-denegerate):

# %%
# Generate a random tensor
rank = 5
A = tf.random.normal(shape=[d, rank])
T = constructSymTensor(components=A)
print(T.shape)

# %% [markdown]
# Test non-degenrate tensor decomposition on this random tensor:

# %%
T_prime = model(T = T, tensor_rank = rank)
print(tf.norm(T_prime - T)/tf.norm(T))

# %% [markdown]
# Now let us see if training changes anything:

# %%
def train_loop(model, T, rank_T, num_epochs = 10):
    epochs = range(num_epochs)
    x = []
    y = []
    for epoch in epochs:
        train(model, T, rank_T)
        x.append(model.x_magic.numpy())
        y.append(model.y_magic.numpy())
        current_loss = loss_func(T, model(T, rank_T))
        print("Epoch {}: x= {}, y= {}, loss= {}".format(epoch, x[-1], y[-1], current_loss))


# %%
train_loop(model, T, rank, num_epochs = 100)

# %% [markdown]
# Non-denegerate case needs not gradient descent. Now we turn to overcomplete case:

# %%
dim = 10
rank = 11
A = tf.random.normal(shape=[dim, rank])
T = constructSymTensor(A)
model_overcomplete = OvercompleteTensorDecomposition(dimension= dim)


# %%
overcomplete_param = rank - dim
def train_overcomplete(model_overcomplete, T, rank, overcomplete_param, learning_rate = 1e-6):
    with tf.GradientTape() as t:
        current_loss = loss_func(T, model_overcomplete(T= T, tensor_rank = rank, overcomplete_param = overcomplete_param))
    dx, dy = t.gradient(current_loss,[model.x_magic, model.y_magic])
    model.x_magic.assign_sub(learning_rate * dx)
    model.y_magic.assign_sub(learning_rate * dy)
def train_loop_overcomplete(model_overcomplete, T, rank, overcomplete_param, num_epochs = 100, learning_rate = 1e-6):
    epochs = range(num_epochs)
    x = []
    y = []
    for epoch in epochs:
        train_overcomplete(model_overcomplete, T, rank, overcomplete_param, learning_rate)
        x.append(model_overcomplete.x_magic.numpy())
        y.append(model_overcomplete.y_magic.numpy())
        current_loss = loss_func(T, model_overcomplete(T, rank_T, overcomplete_param))
        print("Epoch {}: x= {}, y= {}, loss= {}".format(epoch, x[-1], y[-1], current_loss))


# %%
train_loop_overcomplete(model_overcomplete, T, rank, overcomplete_param)

# %% [markdown]
# Remaining cells are for function testings:

# %%
# Check tensor-vector product
a = tf.ones(shape = [2,2,2])
b = tf.ones(shape = [2])
c = tf.tensordot(a,b,axes = [[0],[0]])
print(c)


# %%
# Check tensor-vector product
a = tf.ones(shape = (2,2,2))
b = tf.ones(shape = (2,))
c = tf.tensordot(a,b, axes=1)
print(c)


# %%
b = tf.random.normal(shape = (3,2))
B = constructSymTensor(components=b)
print(B)
print(b[0,0]**3 + b[0,1]**3)


# %%
a = tf.constant(value= [[1,1],[0,-1]],dtype=float)
b = tf.random.normal(shape = [2])
print(a,b)


# %%
print(tf.tensordot(a,b,axes=[[1],[0]]))
print(tf.linalg.diag_part(a))


# %%



