"""
Main .py file to perform overcomplete tensor decomposition
"""
import tensorflow as tf
import numpy as np
import time, collections
from utils import OvercompleteTensorDecomposition, constructSymTensor

def loss_func(T1: tf.Tensor, T2: tf.Tensor):
    """
    relative l2 loss
    """
    return tf.nn.l2_loss(T1-T2)/tf.norm(T1)

def train_loop_overcomplete(model_overcomplete, T, rank, overcomplete_param, learning_rate = 1e-6):
    current_loss, epoch = np.Inf, 0
    times = collections.defaultdict(list)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    while current_loss > 1e-1 and epoch <= 10000:
        epoch += 1
        # print time cost every 500 epochs
        if epoch % 500 == 0:
            with tf.GradientTape() as t:
                print('--'*10)
                print("Forward computation...")
                forward_start_time = time.time()
                T_recovered, Jennrich_times, ls_times = model_overcomplete(T = T, tensor_rank = rank, 
                overcomplete_param = overcomplete_param, timing = True)
                forward_time = time.time() - forward_start_time
                current_loss = loss_func(T, T_recovered)
                print("Time to make forward computation: {0:.5f}s".format(forward_time))
            times['forward'] += [forward_time]
            times['Jennrich1'] += [Jennrich_times[0]]
            times['Jennrich2'] += [Jennrich_times[1]]
            times['ls1'] += [ls_times[0]]
            times['ls2'] += [ls_times[1]]
            print('--'*10)
            print("Computing gradients...")
            grad_start_time = time.time()
            dx, dy = t.gradient(current_loss,[model_overcomplete.x_magic, model_overcomplete.y_magic])
            grad_time = time.time() - grad_start_time
            times['grad'] += [grad_time]
            print("Time to compute gradients: {0:.5f}s".format(grad_time))
        else:
            with tf.GradientTape() as t:
                # forward computation
                T_recovered = model_overcomplete(T= T, tensor_rank = rank, overcomplete_param = overcomplete_param)
                current_loss = loss_func(T, T_recovered)
            # compute gradient
            dx, dy = t.gradient(current_loss,[model_overcomplete.x_magic, model_overcomplete.y_magic])
        
        if tf.norm(dx) <= 1e-3 and tf.norm(dy) <= 1e-3:
            print('--'*10)
            print("Converged since gradients are small. Return...")
            break
        # update parameters
        opt.apply_gradients(zip([dx,dy], [model_overcomplete.x_magic, model_overcomplete.y_magic]))
        # model_overcomplete.x_magic.assign_sub(learning_rate * dx)
        # model_overcomplete.y_magic.assign_sub(learning_rate * dy)

        # print loss and gradient every 500 epochs
        if epoch % 500 == 0:
            T_recovered = model_overcomplete(T= T, tensor_rank = rank, overcomplete_param = overcomplete_param)
            current_loss = loss_func(T, T_recovered)
            print('--'*10)
            print("Epoch {}: x= {}, y= {}, dx = {}, dy = {}, loss= {:.5f}, dx_norm = {:.5f}, dy_norm = {:.5f}".format(epoch, model_overcomplete.x_magic.numpy(), model_overcomplete.y_magic.numpy(), dx.numpy(), dy.numpy, current_loss, tf.norm(dx), tf.norm(dy)))

    return current_loss, epoch, times

if __name__ == "__main__":
    dim = 10
    rank = 11
    A = tf.random.normal(shape=[dim, rank], seed = 12)
    T = constructSymTensor(A)
    # learning_rates = [50,20,10,5]
    learning_rates = [5]
    MAX_REPEAT = 10
    success = []
    for learning_rate in learning_rates:
        for i in range(MAX_REPEAT):
            model_overcomplete = OvercompleteTensorDecomposition(dimension = dim)
            overcomplete_param = rank - dim
            print('--'*10)
            print("Algorithm start with learning rate = {}. Trial {}".format(learning_rate, i+1))
            total_start_time = time.time()
            loss, epoch, times = train_loop_overcomplete(model_overcomplete, T, rank, overcomplete_param,learning_rate= learning_rate)
            total_time = time.time() - total_start_time
            print("Algorithm terminates in {:.5f}s".format(total_time))
            if loss <= 1e-1:
                print("Loss = {}. Successfully decomposed the target tensor in {} epochs!".format(loss, epoch))
                success += [(learning_rate, epoch, total_time)]
            else:
                print("After {} epochs, loss = {} is still too large. Try to restart...".format(epoch, loss))
            print('--'*10)
            for key in times:
                print("Average time in step {}: {}".format(key, np.mean(times[key])))
    print('--'*10)
    print("All successful trials:")
    print(success)