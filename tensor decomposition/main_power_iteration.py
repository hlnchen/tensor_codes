import numpy as np
import time, collections, utils_np

if __name__ == "__main__":
    dim = 10
    rank = 12
    A = np.random.multivariate_normal(mean = np.zeros(dim), cov = np.identity(dim), size = rank).T
    T = utils_np.constructSymTensor(A)
    MAX_REPEAT = 10
    success = []
    for i in range(MAX_REPEAT):
        print('--'*10)
        total_start_time = time.time()
        err, (A_r, Xi_r, A_k, Xi_k) = utils_np.tensorDecomposition(T, rank, timing=True)
        total_time = time.time() - total_start_time
        print("Algorithm terminates in {:.5f}s".format(total_time))
        if err <= 1e-1:
            print("Relative error = {}. Successfully decomposed the target tensor in {} ms!".format(err, total_time))
            success += [(A_r, Xi_r, A_k, Xi_k)]
        else:
            print("After power iterations, relative error = {} is still too large. Try to restart...".format(err))
        print('--'*10)
    print('--'*10)
    print("Number of successful trials = {}".format(len(success)))
    print("Exit....")