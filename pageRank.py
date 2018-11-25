import sys,numpy as np,networkx as nx
from scipy.sparse import csc_matrix



class PageRank:
    def __init__(self, graph):
        self.graph = graph
        self.N = len(self.graph)
        self.d = 0.85
        self.eps = 1.0e-8
        self.ranks = dict()

    def read(fileName):
        G = nx.read_graphml("graph.xml")
        graph = nx.to_numpy_matrix(G)
        return graph
    def to_markov(self):
        A = csc_matrix(self.graph, dtype=np.float)
        col_sum = np.array(A.sum(1))[:, 0]
        A = A.todense()
        n = self.N
        for i in range(0, n):
            if col_sum[i] == 0:
                A[i] = 1 / n
            else:
                A[i] /= col_sum[i]
        return A.transpose()

    def rank(self):
        self.graph = self.to_markov()
        n = self.N
        v = np.random.rand(n, 1)
        v /= np.linalg.norm(v, 1)
        v_p = np.ones((n, 1), dtype=np.float32) * 100
        H = (self.d * self.graph) + (((1 - self.d) / n) * np.ones((n, n), dtype=np.float32))

        v_t = v
        while sum(np.abs(v_t - v_p)) > self.eps:
            v_p = v_t
            v_t = np.matmul(H, v_t) #matrice multiplication
        self.ranks = v_t
        return v_t

    def print(self):
        print(self.ranks)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Expected input format: python pageRank.py <data_filename>')
    else:
        G = PageRank.read(sys.argv[1])
        graph = PageRank(G)
        graph.rank()
        graph.print()
