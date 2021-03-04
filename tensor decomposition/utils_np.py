import time
import numpy as np
from sklearn.preprocessing import normalize

def Jennrich(M1, M2, r: int):
    """
    Inputs:
    M1, M2: matrix of shape [d,d]
    r: rank of M1 and M2
    Outputs:
    the eigenvectors of M1 * M2**(-1)
    """
    U, _, __ = np.linalg.svd(M1)
    W = U[:, 0 : r]
    M1_whitened = W.T @ M1 @ W
    M2_whitened = W.T @ M2 @ W
    M = M1_whitened @ np.linalg.inv(M2_whitened)
    e, P = np.linalg.eig(M)
    return W @ P.real

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
    B = np.linalg.pinv(A_r).T
    Xi = np.diagonal(B.T @ Tx @ B)
    return Xi / np.tensordot(A_r, x, axes= [0,0])
     # Original
    # B = tf.transpose(tf.linalg.pinv(A_r))
    # xi = tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(B), Tx), B))
    # Xi = tf.divide(xi, tf.tensordot(tf.transpose(A_r), x, axes= [[1],[0]]))
    # return Xi

    # TODO: debug this method
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

def decompose(T, x, y, r: int, timing = False):
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
        T_x = np.tensordot(T, x, axes=[[0],[0]])
        T_y = np.tensordot(T, y, axes=[[0],[0]])
        A = Jennrich(T_x, T_y, r = r)
        Xi = matSys(A, T_x, x)
        return A, Xi
    else:
        T_x = np.tensordot(T, x, axes=[[0],[0]])
        T_y = np.tensordot(T, y, axes=[[0],[0]])
        
        Jennrich_start_time = time.time()
        A = Jennrich(T_x, T_y, r = r)
        Jennrich_time = time.time() - Jennrich_start_time
        print("Time to perform Jennrich's algorithm: {0:.5f}s".format(Jennrich_time))

        ls_start_time = time.time()
        Xi = matSys(A, T_x, x)
        ls_time = time.time() - ls_start_time
        print("Time to solve the least squares problem: {0:.5f}s".format(ls_time))
        return A, Xi, Jennrich_time, ls_time
    
def constructSymTensor(components, weights = None):
    """
    # Construct a 3rd order symmetric tensor with columns of components and muliplicative constants from weights
    # Inputs:
    # components: a matrix of shape [d,r]
    # weights: a vector of shape [r, 1]
    # Outputs:
    # T = sum(weight[i] * component[i]@3) (where @ denotes the tensor product)
    """
    _, rank = np.shape(components)
    if weights is None:
        weights = np.ones((1,rank))
    return np.einsum('il,jl,kl->ijk', components * weights, components, components)
    

class OvercompleteTensorDecomposition():
    """
    Overcomplete tensor decomposition
    """
    def __init__(self, dimension, name = None):
        self.x_magic, self.y_magic, self.x_2nd, self.y_2nd = np.random.multivariate_normal(np.zeros(dimension), np.identity(dimension), size = 4)
    def __call__(self, T, tensor_rank, overcomplete_param, timing = False):
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

            return T_first + T_second, (A_r, Xi_r, A_k, Xi_k)
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

            return T_first + T_second, (A_r, Xi_r, A_k, Xi_k), [Jennrich_time1, Jennrich_time2], [ls_time1, ls_time2]

# TODO: debug, check broadcasting dimensions
def tensorPowerIteration(T, k, max_steps = 100):
    """
    Compute the components of T using tensor power iteration
    Inputs:
    T: a symmetric (orthogonal) tensor of shape [d,d,d]
    k: rank of T
    max_steps: maximal step in the iteration
    Outputs:
    A matrix of shape [d,k] containing the components of T in columns
    """
    dim = T.shape[0]
    res, weights = [], []
    for i in range(k):
        # intialization
        x = np.expand_dims(np.random.multivariate_normal(np.zeros(dim), np.identity(dim)),1)
        x /= np.linalg.norm(x)
        # tensor power iterations
        for step in range(max_steps):                                                   # compute T(I, x, x)/||T(I,x,x)||
            x = np.expand_dims(np.tensordot(T, x @ x.T, axes = 2),1)
            x /= np.linalg.norm(x)
        # update & deflation
        w = x.T @ np.tensordot(T, x @ x.T, axes = 2)                                    # get the factor
        res.append(x[:,0])
        weights.append(w)
        T -= constructSymTensor(components = x, weights = w)                            # deflation
    return np.array(res).T, np.array(weights).T                                         # return as a matrix

def tensorTransform(T, A, B, C):
    """
    Compute T[A,B,C]
    """
    return np.einsum('mln,im,jl,kn->ijk', T, A, B, C)

def tensorDecomposition(T, rank, timing = False):
    """
    TODO: check why explosion happens
    1. correctness of the transformation function               CHECKED
    2. correctness of the orthogonalizer                        
    3. correctness of the power iteration                       CHECKED
    4. correctness of the alternation
    """
    dim, err, MAX_TRIAL = T.shape[0], 1, 1000
    # search for a good intialization
    step = 0
    while err > 1e-1 and step < MAX_TRIAL:
        roughDecomposer = OvercompleteTensorDecomposition(dim)
        T_recovered, (A_r, Xi_r, A_k, Xi_k) = roughDecomposer(T, rank, rank-dim)
        err = np.linalg.norm(T-T_recovered)/np.linalg.norm(T)
        step += 1
    # alternatively do tensor power iteration
    step = 0
    while err > 1e-2 and step < MAX_TRIAL:
        # tensor power iteration on subproblem #1 (rank-dim components)
        if timing:
            pwr1_start = time.time()
        P = np.linalg.pinv(A_k)
        T2 = constructSymTensor(A_r, Xi_r)
        T1 = tensorTransform(T-T2, P, P, P)
        A1, Xi_k = tensorPowerIteration(T1, rank - dim)
        # A_k = normalize(A_k @ A1, axis = 0)
        A_k = A_k @ A1
        if timing:
            pwr1_time = time.time() - pwr1_start

        # tensor power iteration on subproblem #2 (dim components)
        if timing:
            pwr2_start = time.time()
        Q = np.linalg.pinv(A_r)                                                 # build an orthogonalizer for A_r
        T1 = constructSymTensor(A_k, Xi_k)
        T2 = tensorTransform(T - T1, Q, Q, Q)
        A2, Xi_r = tensorPowerIteration(T2, dim)
        # A_r = normalize(A_r @ A2, axis = 0)
        A_r = A_r @ A2
        if timing:
            pwr2_time = time.time() - pwr2_start 
        
        # reconstruction
        T_recovered = constructSymTensor(A_r, Xi_r) + T1
        err = np.linalg.norm(T - T_recovered)
        if step % 10 == 0:
            print('Step {}: err = {:.5f}'.format(step, err))
            if timing:
                print('Time of: 1st subproblem {:.5f}, 2nd subproblem {:.5f}'.format(pwr1_time, pwr2_time))
        step += 1
    return err, (A_r, Xi_r, A_k, Xi_k)
