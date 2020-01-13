# coding=utf-8
import tensorflow as tf

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = tf.concat([source,target], axis=0)
    double_size = int(total.shape[0])
    for i in range(0,double_size-1):
        if i == 0:
            total0 = tf.concat([tf.expand_dims(total,0),tf.expand_dims(total,0)], axis=0)
        else:
            total0 =tf.concat([total0,tf.expand_dims(total,0)], axis=0)

    for i in range(0,double_size-1):
        if i == 0:
            total1 = tf.concat([tf.expand_dims(total,1),tf.expand_dims(total,1)], axis=1)
        else:
            total1 =tf.concat([total1,tf.expand_dims(total,1)], axis=1)

    L2_distance = tf.reduce_sum((total0 - total1) ** 2, 2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size_1 = int(source.shape[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    print('batch_size_1'+str(batch_size_1))
    XX = kernels[:batch_size_1, :batch_size_1]
    YY = kernels[batch_size_1:, batch_size_1:]
    XY = kernels[:batch_size_1, batch_size_1:]
    YX = kernels[batch_size_1:, :batch_size_1]

    loss = tf.reduce_mean(XX + YY - XY -YX)
    return loss