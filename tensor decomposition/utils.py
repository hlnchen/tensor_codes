import tensorflow as tf
import numpy as np
import time

def Jennrich(M1: tf.Tensor, M2: tf.Tensor, r: int):
    """
    Inputs:
    M1, M2: matrix of shape [d,d] of the same rank = r
    Outputs:
    the eigenvectors of M1 * M2**(-1)
    """
    S, U, V = tf.linalg.svd(M1)
    W = U[:, 0 : r]
    M1_whitened = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(W), M1), W)
    M2_whitened = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(W), M2), W)
    M = tf.linalg.matmul(M1_whitened, tf.linalg.inv(M2_whitened))
    e, P = tf.linalg.eig(M)
    return tf.linalg.matmul(W, tf.math.real(P))

def matSys(A_r, Tx, x):
    """
    Solve a matrix valued linear system A_r * Xi * diag(A_r.T * x) * A_r.T = Tx
    Inputs: 
    A_r: [d,r]; 
    Tx: [d,d]; 
    x: [d,1]
    Outputs:
    Xi: [r,1]
    """
    # Original
    # B = tf.transpose(tf.linalg.pinv(A_r))
    # xi = tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(B), Tx), B))
    # Xi = tf.divide(xi, tf.tensordot(tf.transpose(A_r), x, axes= [[1],[0]]))
    # return Xi

    B = tf.transpose(tf.linalg.pinv(A_r))
    Xi = tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(B), Tx), B))
    return Xi / tf.tensordot(A_r, x, axes= [0,0])

    # With Khatri-Rao and without pinv
    # _, r = tf.shape(A_r)
    # A = tf.reshape(tf.expand_dims(A_r,1) * tf.expand_dims(A_r,0), [-1, r])
    # Xi = tf.linalg.lstsq(A, tf.reshape(Tx,[-1,1]))
    # return Xi/tf.tensordot(A_r, x, axes = [0,0])

    # # With Khatri-Rao
    # _, r = tf.shape(A_r)
    # A = tf.reshape(tf.expand_dims(A_r,1) * tf.expand_dims(A_r,0), [-1, r])
    # Xi = tf.matmul(tf.linalg.pinv(A), tf.reshape(Tx,[-1,1]))
    # return Xi/tf.tensordot(A_r, x, axes = [0,0])

def decompose(T: tf.Tensor, x: tf.Tensor, y: tf.Tensor, r: int, timing = False):
    """
    Given an input tensor, run Jennrich's algorithm and length recovery
    Inputs:
    T: tensor to be decomposed, of shape [d,d,d];
    x,y: vectors to generate two inputs for Jennrich(), of shape [d,1]; 
    r: rank of T.
    timing: a flag to control if to print computation time
    Outputs:
    A: a matrix with columns as directions of tensor components, of shape [d,r]
    Xi: a vector containing the length of each conponent, of shape [r,1]
    """
    if not timing:
        T_x = tf.tensordot(T, x, axes=[[0],[0]])
        T_y = tf.tensordot(T, y, axes=[[0],[0]])
        A = Jennrich(T_x, T_y, r = r)
        Xi = matSys(A, T_x, x)
        return A, Xi
    else:
        T_x = tf.tensordot(T, x, axes=[[0],[0]])
        T_y = tf.tensordot(T, y, axes=[[0],[0]])
        
        Jennrich_start_time = time.time()
        A = Jennrich(T_x, T_y, r = r)
        Jennrich_time = time.time() - Jennrich_start_time
        print("Time to perform Jennrich's algorithm: {0:.5f}s".format(Jennrich_time))

        ls_start_time = time.time()
        Xi = matSys(A, T_x, x)
        ls_time = time.time() - ls_start_time
        print("Time to solve the least squares problem: {0:.5f}s".format(ls_time))
        return A, Xi, Jennrich_time, ls_time
    

def constructSymTensor(components: tf.Tensor, weights = None):
    """
    # Construct a 3rd order symmetric tensor with columns of components and muliplicative constants from weights
    # Inputs:
    # components: a matrix of shape [d,r]
    # weights: a vector of shape [r, 1]
    # Outputs:
    # T = sum(weight[i] * component[i]@3) (where @ denotes the tensor product)
    """
    _, rank = tf.shape(components)
    if weights == None:
        weights = tf.ones(shape=[rank])
    return tf.einsum('il,jl,kl->ijk', components * weights, components, components)
    # T = tf.zeros(shape = [dim,dim,dim])
    # for i in range(rank):
    #     ai = tf.expand_dims(components[:,i], axis = 1)
    #     T += tf.einsum('ij,kl->ijk', tf.tensordot(ai, weights[i,0] * ai, axes= [[1],[1]]) , ai)
    # return T

    # components_aux = []
    # for i in range(rank):
    #     a = tf.expand_dims(components[:,i], -1)
    #     components_aux.append(tf.reshape(tf.matmul( weights[i] * a, tf.transpose(a)), [-1]))
    # components_aux = tf.transpose(tf.stack(components_aux))
    # return tf.reshape(tf.matmul(components_aux, tf.transpose(components)), [dim,dim,dim])


class OvercompleteTensorDecomposition(tf.Module):
    """
    Overcomplete tensor decomposition module implemented in TensorFlow
    """
    def __init__(self, dimension, name = None):
        super().__init__(name = name)
        init = tf.random_normal_initializer()
        init1 = init(shape = [dimension], dtype = tf.float32)
        init2 = init(shape = [dimension], dtype = tf.float32)
        self.x_magic = tf.Variable(initial_value = tf.divide(init1,tf.norm(init1)), name = "x_magic")
        self.y_magic = tf.Variable(initial_value = tf.divide(init2,tf.norm(init2)), name = "y_magic")
        self.x_2nd = tf.constant(value= init(shape = [dimension], dtype= tf.float32), name = "x_2nd")
        self.x_2nd /= tf.norm(self.x_2nd)
        self.y_2nd = tf.constant(value= init(shape = [dimension], dtype= tf.float32), name = "y_2nd")
        self.y_2nd /= tf.norm(self.y_2nd)
    def __call__(self, T: tf.Tensor, tensor_rank, overcomplete_param, timing = False):
        """
        Inputs:
        T: a symmetric tensor of shape [d,d,d]
        tensor_rank: rank of T
        overcomplete_param: any (d - overcomplete_param) in T will be linearly independent. 
                            Set to tensor_rank - d for random tensors.
        timing: a flag to control if to print computation time for each step
        """
        # x = self.x_magic/tf.norm(self.x_magic)
        # y = self.y_magic/tf.norm(self.y_magic)
        if not timing:
            # Decompose the first r compoents
            A_r, Xi_r = decompose(T, self.x_magic, self.y_magic, r = tensor_rank - overcomplete_param)

            # Deflation
            T_first = constructSymTensor(components= A_r, weights= Xi_r)
            R = T - T_first

            # 2nd decomposition
            A_k, Xi_k = decompose(R, self.x_2nd, self.y_2nd, r = overcomplete_param)
            
            # Reconstruction
            T_second = constructSymTensor(components= A_k, weights= Xi_k)

            return T_first + T_second
        else:
            # Decompose the first r compoents
            print("First decomposition...")
            first_start_time = time.time()
            A_r, Xi_r, Jennrich_time1, ls_time1 = decompose(T, self.x_magic, self.y_magic, r = tensor_rank - overcomplete_param, timing = True)

            # Deflation
            print("First reconstruction...")
            first_recon_start = time.time()
            T_first = constructSymTensor(components= A_r, weights= Xi_r)
            R = T - T_first
            print("First reconstruction time: {:.5f}".format(time.time() - first_recon_start))

            # 2nd decomposition
            print("Second decomposition...")
            A_k, Xi_k, Jennrich_time2, ls_time2 = decompose(R, self.x_2nd, self.y_2nd, r = overcomplete_param, timing = True)

            # Reconstruction
            print("Second reconstruction...")
            second_recon_start = time.time()
            T_second = constructSymTensor(components= A_k, weights= Xi_k)
            print("Second reconstruction time: {:.5f}".format(time.time() - second_recon_start))

            return T_first + T_second, [Jennrich_time1, Jennrich_time2], [ls_time1, ls_time2]