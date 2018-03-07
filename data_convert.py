import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import os
import networkx as nx
from sklearn import preprocessing
import json
from networkx.readwrite import json_graph

import utils_gcn
import utils_sage

from my_encoder import my_encoder


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pkl.dump(obj, output, pkl.HIGHEST_PROTOCOL)


"""
GCN Format: 

In order to use your own data, you have to provide

an N by N adjacency matrix (N is the number of nodes),
an N by D feature matrix (D is the number of features per node), and
an N by E binary label matrix (E is the number of classes).
Have a look at the load_data() function in utils.py for an example.
~~~~~~~~~~~~~~~~~~~~~~~
The input to the inductive model contains:

x, the feature vectors of the labeled training instances,
y, the one-hot labels of the labeled training instances,
allx, the feature vectors of both labeled and unlabeled training instances (a superset of x),
graph, a dict in the format {index: [index_of_neighbor_nodes]}.
Let n be the number of both labeled and unlabeled training instances. These n instances should be indexed from 0 to n - 1 in graph with the same order as in allx.
~~~~~~~~~~~~~~~~~~~~~~~
In addition to x, y, allx, and graph as described above, the preprocessed datasets also include:

tx, the feature vectors of the test instances,
ty, the one-hot labels of the test instances,
test.index, the indices of test instances in graph, for the inductive setting,
ally, the labels for instances in allx.
The indices of test instances in graph for the transductive setting are from #x to #x + #tx - 1, with the same order as in tx.

You can use cPickle.load(open(filename)) to load the numpy/scipy objects x, y, tx, ty, allx, ally, and graph. test.index is stored as a text file.
"""


def sage_format_to_GCN_format(num_of_x=1000, sage_prefix='ppi', root_dir='data'):
    train_data = utils_sage.load_data(sage_prefix, root_dir='data')
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]

    print('num of nodes: {}'.format(len(G.node)))
    print('num of features: {}'.format(len(features)))
    print('num of ids: {}'.format(len(id_map)))
    print('num of classes: {}'.format(len(class_map)))

    print('data file is read successfully')

    test_index = []
    all_x_index = []
    labels = {}

    print('splitting data...')

    for key, value in id_map.items():
        labels.update({value: class_map[key]})

    indexed_labels = []
    size = len(labels)

    print("num of nodes: {}".format(size))

    for i in range(size):
        if i in labels:
            indexed_labels.append(labels[i])
        else:
            print("have not label for node {}".format(i))
            indexed_labels.append(-1)

    if not isinstance(indexed_labels[0], int):
        print("Not support multi class")
        return

    set_labels = list(set(indexed_labels))

    ally = get_one_hot(indexed_labels, len(set_labels))

    # ======================
    vertices = G.node

    for key, value in vertices.items():
        if value['test']:
            test_index.append(id_map[key])
        # else:
        #     all_x_index.append(id_map[key])

    for i in range(size):
        if i not in test_index:
            all_x_index.append(i)

    x_index = np.array(np.random.choice(
        len(all_x_index), num_of_x, replace=False))
    x_index = np.array(all_x_index)[x_index]
    x = sp.csr_matrix(features[x_index])
    y = ally[x_index]

    ty = ally[test_index]
    tx = sp.csr_matrix(features[test_index])

    allx = sp.csr_matrix(features[all_x_index])
    ally = ally[all_x_index]

    H = nx.relabel_nodes(G, id_map)
    graph = nx.to_dict_of_lists(H)

    # root_dir = 'tmp'
    save_object(allx, "{}/ind.{}.allx".format(root_dir, sage_prefix))
    save_object(ally, "{}/ind.{}.ally".format(root_dir, sage_prefix))
    save_object(graph, "{}/ind.{}.graph".format(root_dir, sage_prefix))
    save_object(tx, "{}/ind.{}.tx".format(root_dir, sage_prefix))
    save_object(ty, "{}/ind.{}.ty".format(root_dir, sage_prefix))
    save_object(x, "{}/ind.{}.x".format(root_dir, sage_prefix))
    save_object(y, "{}/ind.{}.y".format(root_dir, sage_prefix))

    np.savetxt("{}/ind.{}.test.index".format(root_dir, sage_prefix),
               test_index, delimiter='\n', fmt='%s')
    # save_object(test_index, "data/ind.{}.test.index".format(sage_prefix))
    print('Done')


"""
Sage Format:
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

<train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
<train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
<train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
<train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
<train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)
To run the model on a new dataset, you need to make data files in the format described above. To run random walks for the unsupervised model and to generate the -walks.txt file) you can use the run_walks function in graphsage.utils.
"""


def GCN_format_to_sage_format(data_name='citeseer', root_dir='data'):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = utils_gcn.load_data(
        data_name, root_dir=root_dir)

    G = nx.from_scipy_sparse_matrix(adj)
    train_index = np.where(train_mask)[0]
    val_index = np.where(val_mask)[0]
    test_index = np.where(test_mask)[0]
    y = y_train + y_val + y_test
    y = np.argmax(y, axis=1)

    for i in range(len(y)):
        if i in val_index:
            G.node[i]['val'] = True
            G.node[i]['test'] = False
            G.node[i]['train'] = False
        elif i in test_index:
            G.node[i]['test'] = True
            G.node[i]['val'] = False
            G.node[i]['train'] = False
        elif i in train_index:
            G.node[i]['test'] = False
            G.node[i]['val'] = False
            G.node[i]['train'] = True
        else:
            G.node[i]['test'] = False
            G.node[i]['val'] = False
            G.node[i]['train'] = False

    data = json_graph.node_link_data(G)
    with open("{}/{}-G.json".format(root_dir, data_name), "w", encoding="utf8") as f:
        json.dump(data, f, cls=my_encoder)
    classMap = {}
    idMap = {}
    for i in range(len(y)):
        classMap[i] = y[i]
        idMap[i] = i
    with open("{}/{}-id_map.json".format(root_dir, data_name), "w", encoding="utf8") as f:
        json.dump(idMap, f, cls=my_encoder)
    with open("{}/{}-class_map.json".format(root_dir, data_name), "w", encoding="utf8") as f:
        json.dump(classMap, f, cls=my_encoder)
    np.save(open("{}/{}-feats.npy".format(root_dir, data_name), "wb"),
            features.todense())


"""
node2vec Format:
Input

The supported input format is an edgelist:

node1_id_int node2_id_int <weight_float, optional>
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

Output

The output file has n+1 lines for a graph with n vertices. The first line has the following format:

num_of_nodes dim_of_representation
The next n lines are as follows:

node_id dim1 dim2 ... dimd
where dim1, ... , dimd is the d-dimensional representation learned by node2vec.
"""


def sage_format_to_node2vec_format(sage_prefix='ppi', root_dir='data'):
    train_data = utils_sage.load_data(sage_prefix, root_dir='data')
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]

    try:
        all_labels = np.fromiter(class_map.values(), dtype=int)
    except Exception:
        print("Not support multiple classes")
        return

    graph = nx.to_dict_of_lists(G)

    edges = []

    for key, value in graph.items():
        if isinstance(value, int):
            edges.append([key, value])
        else:
            for v in value:
                edges.append([key, v])

    np.savetxt("{}/{}.edgelist".format(root_dir, sage_prefix),
               edges, delimiter=' ', fmt='%s')
    return None


if __name__ == '__main__':
    # sage_format_to_GCN_format(sage_prefix='ppi', root_dir='data')
    GCN_format_to_sage_format(data_name='pubmed', root_dir='data')
    # sage_format_to_node2vec_format(sage_prefix='citeseer', root_dir='data')
