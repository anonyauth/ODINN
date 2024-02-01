import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
conv1d = tf.layers.conv1d


def DeGroot(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('dg'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False, activation='relu')
        seq_fts = tf.layers.conv1d(seq_fts, output_dim, 1, use_bias=False, activation='relu')

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 10
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)

        alpha = tf.Variable(np.ones(hops),trainable=True,dtype=tf.float32)
        alpha = tf.nn.relu(alpha)

        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        subspace = list()
        subspace.append(H1)
        agg = tf.math.multiply(alpha[0],H1)
        for i in range(hops-1):
            Hi = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            subspace.append(Hi)
            agg += tf.math.multiply(alpha[i+1],Hi)
        
        vals = tf.expand_dims(agg, axis=0)

        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)


def Friedkin_Johnsen(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('fj'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False, activation="relu")
        seq_fts = tf.layers.conv1d(seq_fts, output_dim, 1, use_bias=False, activation='relu')

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 10
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)
        m = int(m)
        
        stub = tf.Variable(np.ones(hops),trainable=True,dtype=tf.float32)
        stub = tf.sigmoid(stub)
        
        H0 = tf.multiply(stub[0],seq_fts)
        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        H1 = tf.multiply(tf.constant(1.0)-stub[0],H1)
        subspace = list()
        subspace.append(H1+H0)
        for i in range(hops-1):
            Hi1 = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            Hi1 = tf.multiply(tf.constant(1.0)-stub[i+1],Hi1)
            Hi0 = tf.multiply(stub[i+1],subspace[i])
            subspace.append(Hi1+Hi0)
        H =  subspace[-1]
        vals = tf.expand_dims(H, axis=0)

        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)


def linear_layer(inputs, output_dim, activation, in_drop=0.0):
    with tf.name_scope('ll'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = seq_fts
        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias2")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)

