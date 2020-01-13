# coding=utf-8
import pickle
import numpy as np
import tensorflow as tf


def read_pkl():
    f = open(r'./10_128_py2.pkl','rb')
    a = pickle.load(f)
    f.close()
    return a

def pedcc_fc(x):
    map_dict = read_pkl()
    tmp_list = [y.numpy() for y in map_dict.values()]
    label_40D_tensor = np.concatenate(tmp_list).transpose()
    w = tf.constant(label_40D_tensor)
    w = tf.cast(w, dtype=tf.float32)
    ww = tf.nn.l2_normalize(w, axis=0, epsilon=1e-10)
    wlen = tf.norm(ww, axis=0, keepdims=True)
    xlen = tf.norm(x, axis=1, keepdims=True)

    # print(x)
    # print(ww)
    cos_theta = tf.matmul(x, ww)
    # print(cos_theta
    cos_theta = cos_theta / tf.reshape(xlen, [-1,1]) / tf.reshape(wlen, [1, -1])
    cos_theta =tf.clip_by_value(cos_theta, -1, 1)
    cos_theta = tf.multiply(cos_theta, tf.reshape(xlen, [-1,1]))
    # print(cos_theta)

    return cos_theta

def AM_logits_compute(cos_theta, label_batch):
    m = 0.35
    s = 7.5
    phi = cos_theta - m
    label_onehot = tf.one_hot(label_batch, 10)  ##10类
    adjust_theta = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)
    return adjust_theta

def AM_loss(input, target):
    AM_logits = AM_logits_compute(input, target)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=AM_logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    am_loss = tf.add_n([cross_entropy_mean], name='total_loss')
    return am_loss
