import math, random
import collections
import pandas as pd
import numpy as np

class Louvain:

    def __init__(self):
        pass

    def construct_graph(self, nodes, edges):  # construct the graph
        self.nodes, self.edges = nodes, edges
        self.m = 0    # sum of the weights of all links in network
        self.k_i = [0 for n in nodes]   # sum of the weights of the links incident to node i
        self.edges_of_node = {}
        self.w = [0 for n in nodes]
        for e in edges:
            self.m += e[1]
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1] if e[0][1] != e[0][0] else 0
            if e[0][0] not in self.edges_of_node: self.edges_of_node[e[0][0]] = [e]
            else: self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node: self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]: self.edges_of_node[e[0][1]].append(e)
        self.communities = [n for n in nodes] # access community of a node in O(1) time
        self.actual_partition = []

    def load_data(self, path): # load data from csv file and construct the graph
        data = pd.read_csv(path)
        G_ = np.array(data.values.tolist())  # the graph
        m = G_.shape[0]   # number of edges
        self.n = np.max(G_)+1    # number of nodes
        print("{} nodes, {} edges".format(self.n, m))
        nodes_ = np.arange(self.n)
        edges_ = [((G_[i,0], G_[i,1]), 1) for i in range(m)]
        return self.construct_graph(nodes_, edges_)

    def best_partition(self):  # find the best partition
        network = (self.nodes, self.edges)
        self.best_q = -1
        i = 1
        while True:
            if not i%100: print("Iter: {}, Q: {}".format(i, self.best_q))
            i += 1
            partition = self.first_phase(network)
            q = self.compute_modularity(partition)
            partition = [c for c in partition if c]
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition
            if q <= self.best_q:
                break
            network = self.second_phase(network, partition)
            self.best_q = max(q, self.best_q)
        print("There are {} communities with best_Q={}, curr_Q={}".format(len(self.actual_partition), self.best_q, q))
        print([len(i) for i in self.actual_partition])
        return (self.actual_partition, self.best_q)

    def compute_modularity(self, partition):  # compute the modularity
        q = 0
        m2 = self.m * 2
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q

    def first_phase(self, network):  # fisrt phase, return the partition
        best_partition = [[node] for node in network[0]]
        self.s_in = [0 for node in network[0]]  # in-links number of a community 
        self.s_tot = [self.k_i[node] for node in network[0]]
        for e in network[1]:
            if e[0][0] == e[0][1]: # only self-loops
                self.s_in[e[0][0]] += e[1]
                self.s_in[e[0][1]] += e[1]
        while True:
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]
                best_community = node_community
                best_gain = 0
                best_partition[node_community].remove(node)
                best_shared_links = 0
                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]: continue   # self loop
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or \
                         e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_links += e[1]
                self.s_in[node_community] -= 2 * (best_shared_links + self.w[node])
                self.s_tot[node_community] -= self.k_i[node]
                self.communities[node] = -1
                communities = {} # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):
                    community = self.communities[neighbor]
                    if community in communities: continue
                    communities[community] = 1
                    shared_links = 0
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]: continue
                        if e[0][0] == node and self.communities[e[0][1]] == community or \
                            e[0][1] == node and self.communities[e[0][0]] == community:
                            shared_links += e[1]
                    # compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = 2 * shared_links - self.s_tot[community] * self.k_i[node] / self.m
                    if gain > best_gain:
                        best_community = community
                        best_gain = gain
                        best_shared_links = shared_links
                # insert _node into the community maximizing the modularity gain
                best_partition[best_community].append(node)
                self.communities[node] = best_community
                self.s_in[best_community] += 2 * (best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                if node_community != best_community:
                    improvement = 1
            if not improvement:
                break
        return best_partition

    def get_neighbors(self, node):  # get neighbors of a node as a generator
        for e in self.edges_of_node[node]:
            if e[0][0] == e[0][1]: continue
            if e[0][0] == node: yield e[0][1]
            if e[0][1] == node: yield e[0][0]

    def second_phase(self, network, partition):  # second phase, return the partition
        nodes_ = [i for i in range(len(partition))]
        # relabelling communities
        communities_ = []
        d = {}
        i = 0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                i += 1
        self.communities = communities_
        # building relabelled edges
        edges_ = {}
        for e in network[1]:
            ci = self.communities[e[0][0]]
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] += e[1]
            except KeyError:
                edges_[(ci, cj)] = e[1]
        edges_ = [(k, v) for k, v in edges_.items()]
        # recomputing k_i vector and storing edges by node
        self.k_i = [0 for n in nodes_]
        self.edges_of_node = {}
        self.w = [0 for n in nodes_]
        for e in edges_:
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1]
            if e[0][0] == e[0][1]:
                self.w[e[0][0]] += e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # resetting communities
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)

    def find_community(self, path_labels):  # find the community and save the result
        # compute the best partition
        print("****************** Partitioning ******************")
        partition_array, q = self.best_partition()
        print("****************** Transforming ******************")
        partition_mat = self.label(partition_array, path_labels)
        # print(partition_mat)
        print("******************    Saving    ******************")
        res = pd.DataFrame(partition_mat, columns=['id', 'category'])
        res.to_csv("result.csv", index=False)
        return partition_mat
 
    def label(self, partition_array, path_labels):  # return the clustered community partition matrix
        partition_dict = collections.defaultdict(int)
        for community in range(len(partition_array)):
            for node in partition_array[community]:
                partition_dict[node] = community
        partition_mat = np.mat(list(partition_dict.items()))
        data = pd.read_csv(path_labels)
        labels = np.array(data.values.tolist())
        n_label = np.max(labels[:,1])
        community_number = np.max(partition_mat[:,1]) + 1
        to_label = [collections.defaultdict(int) for _ in range(community_number)]

        for i in range(labels.shape[0]):
            community = partition_dict[labels[i,0]]
            to_label[community][labels[i,1]] += 1

        transform = collections.defaultdict(int)  # the dict to make community belong to given five clusters
        for j in range(community_number):
            vote = np.mat(list(to_label[j].items())) # j-th community's voting matrix
            if vote.shape[1]!=0: c = vote[np.argmax(vote[:,1]), 0]  
            else: 
                np.random.seed(521)
                c = np.random.randint(5)
                print("{} to {}".format(j, c))
            transform[j] = c

        for k in range(self.n):
            partition_mat[k, 1] = transform[partition_mat[k, 1]]
        partition_mat = np.mat(sorted([(partition_mat[i,0],partition_mat[i,1]) \
            for i in range(partition_mat.shape[0])], key=lambda x: x[0]))
        return partition_mat
     
    def test(self, partition_mat, path_reference, acc):  # Test the accuracy according to 
        print("******************   Testing    ******************")
        n = partition_mat.shape[0]
        data = pd.read_csv(path_reference)
        reference = np.array(data.values.tolist())
        diff = np.sum((reference!=partition_mat)[:,1])
        acc_pseudo = 1 - diff/n
        acc_min = acc_pseudo - 1 + acc
        acc_max = acc_pseudo + 1 - acc
        print("Pseudo Accuracy: {} \t Accuracy Range: [{}, {}]".format(acc_pseudo, acc_min, acc_max))
        return acc_pseudo, acc_min

if __name__ == '__main__':
    path_edges = 'data/edges.csv'
    path_labels = 'data/ground_truth.csv'
    path_reference = 'data/086284.csv'
    acc = 0.86284
    
    louvain = Louvain()
    louvain.load_data(path_edges)
    partition_mat = louvain.find_community(path_labels)
    louvain.test(partition_mat, path_reference, acc)