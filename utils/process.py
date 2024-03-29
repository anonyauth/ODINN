import os
import re
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from utils.inout import load_dataset
from utils.ioo import load_dataset as load_ds
from sklearn.preprocessing import OneHotEncoder


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_split(dataset_str, train_size, shuffle=True):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    labels = np.vstack((ally, ty))

    if dataset_str.startswith('nell'):
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/planetoid/{}.features.npz".format(dataset_str)):
            # print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                            dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended, dtype=np.float32)
            # print("Done!")
            save_sparse_csr("data/planetoid/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/planetoid/{}.features.npz".format(dataset_str))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    global all_labels
    all_labels = labels.copy()

    # split the data set
    idx = np.arange(len(labels))
    no_class = labels.shape[1]  # number of class
    tr_size =train_size
    train_size = [train_size for i in range(labels.shape[1])]
    if shuffle:
        np.random.shuffle(idx)

    idx_train = []
    count = [0 for i in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(no_class):
            if labels[i, j] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    validation_size = (len(labels) - tr_size * no_class) // 10
    idx_val = idx[next:next+validation_size]
    assert next+validation_size < len(idx)
    idx_test = idx[next+validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    size_of_each_class = np.sum(labels[idx_train], axis=0)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



def load_data_sparse_graph(dataset_str, train_size, shuffle=True):

    graph = load_dataset(dataset_str)
    adj = graph.adj_matrix
    features = graph.attr_matrix.tolil()
    labels = graph.labels
    label = labels
    labels = labels.reshape(len(labels), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    labels = onehot_encoder.fit_transform(labels)

    # split the data set
    idx = np.arange(len(labels))
    no_class = labels.shape[1]  # number of class
    tr_size =train_size
    train_size = [train_size for i in range(labels.shape[1])]
    if shuffle:
        np.random.shuffle(idx)
    #print(idx)
    idx_train = []
    count = [0 for i in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(no_class):
            if labels[i, j] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    validation_size = (len(labels) - tr_size * no_class) // 10
    idx_val = idx[next:next+validation_size]
    assert next+validation_size < len(idx)
    idx_test = idx[next+validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_sparse_graph_ms(dataset_str, train_size, shuffle=True):
    
    graph = load_ds(dataset_str)
    adj = graph.adj_matrix
    features = graph.attr_matrix.tolil()
    labels = graph.labels
    labels = labels.reshape(len(labels), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    labels = onehot_encoder.fit_transform(labels)
    
    # print(labels)
    # print(type(features))
    # ss=check_symmetric(adj.toarray())
    # print(ss)#False
    # print(type(labels))


    # split the data set
    idx = np.arange(len(labels))
    no_class = labels.shape[1]  # number of class
    tr_size =train_size
    train_size = [train_size for i in range(labels.shape[1])]
    if shuffle:
        np.random.shuffle(idx)
    #print(idx)
    idx_train = []
    count = [0 for i in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(no_class):
            if labels[i, j] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    validation_size = (len(labels) - tr_size * no_class) // 10
    idx_val = idx[next:next+validation_size]
    assert next+validation_size < len(idx)
    idx_test = idx[next+validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_rw(adj):
    """Random walk transition matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).tocoo()


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj.toarray()
    ss=check_symmetric(adj)
    if not ss:
        adj += adj.T
    adj = sp.csr_matrix(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_rw(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj.toarray()
    ss=check_symmetric(adj)
    if not ss:
        adj += adj.T
    adj = sp.csr_matrix(adj)
    # adj_normalized = normalize_adj_rw(adj)
    adj_normalized = normalize_adj_rw(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


