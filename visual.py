import pandas as pd
import numpy as np
import collections

import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx


class Louvain:

    def __init__(self):
        pass

    def construct_graph(self, path):
        data = pd.read_csv(path)
        self.G_ = np.array(data.values.tolist())  # the graph
        self.m = self.G_.shape[0]   # number of edges
        self.n = np.max(self.G_)    # number of nodes
        # G = nx.karate_club_graph()
        self.G = nx.Graph(directed=True)
        self.G.add_nodes_from(range(self.n))
        self.G.add_edges_from(self.G_)

    def gd(self, partition_dict, path_labels): # return the clustered community partition matrix
        self.n = len(partition_dict)   # number of node
        self.partition_mat = np.mat(list(partition_dict.items()))
        data = pd.read_csv(path_labels)
        labels = np.array(data.values.tolist())
        community_number = np.max(self.partition_mat[:,1]) + 1
        to_label = [collections.defaultdict(int) for _ in range(community_number)]

        for i in range(labels.shape[0]):
            community = partition_dict[labels[i,0]]
            to_label[community][labels[i,1]] += 1

        
        transform = collections.defaultdict(int)  # the dict to make community belong to given five clusters
        for j in range(community_number):
            vote = np.mat(list(to_label[j].items())) # j-th community's voting matrix
            c = vote[np.argmax(vote[:,1]), 0] if vote.shape[1]!=0 else 0
            transform[j] = c

        for k in range(self.n):
            self.partition_mat[k, 1] = transform[self.partition_mat[k, 1]]
        return self.partition_mat

    def find_community(self, path_labels):
        # compute the best partition
        print("****************** Partitioning ******************")
        partition = community_louvain.best_partition(self.G)
        print("****************** Transforming ******************")
        part = self.gd(partition, path_labels)
        print("******************    Saving    ******************")
        res = pd.DataFrame(part, columns=['id', 'category'])
        res.to_csv("data/result.csv", index=False)
        return partition
        
    def test(self, path_reference, acc):
        print("******************   Testing    ******************")
        data = pd.read_csv(path_reference)
        reference = np.array(data.values.tolist())
        same = np.sum(reference!=self.partition_mat)
        acc_pseudo = 1 - same/self.n
        acc_min = acc_pseudo - 1 + acc
        acc_max = acc_pseudo + 1 - acc
        print("Pseudo Accuracy: {} \t Accuracy Range: [{}, {}]".format(acc_pseudo, acc_min, acc_max))
        return acc_pseudo, acc_min

    def draw(self, partition):
        # draw the graph
        pos = nx.spring_layout(self.G)
        # color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(self.G, pos, partition.keys(), node_size=40,
                            cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        plt.savefig("graph.jpg")
        plt.show()

if __name__=="__main__":
    path_edges = 'data/edges.csv'
    path_labels = 'data/ground_truth.csv'
    path_reference = 'data/087194.csv'
    acc = 0.87194


    model = Louvain()
    model.construct_graph(path_edges)
    par = model.find_community(path_labels)
    ac, _ = model.test(path_reference, acc)
    model.draw(par)   # Delete if no need for visualization