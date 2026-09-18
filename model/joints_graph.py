import numpy as np


def get_dimused_Adj(adjacency):
    # adjacency = np.zeros((33, 33))
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    B = adjacency[:, dim_used]
    final_adj = B[dim_used, :]
    return final_adj


def norm_Adjcency(adjacency):
    Dl = np.sum(adjacency, 0)
    print(Dl)
    num_node = adjacency.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, adjacency), Dn)
    return DAD


def get_Adjcency(num_node):
    self_link = [(i, i) for i in range(num_node)]
    neighbor_link_ = [(0, 2), (0, 7), (0, 12), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9),
                      (9, 10), (12, 13), (13, 14), (13, 17), (13, 25),
                      (14, 15), (16, 17), (17, 18), (18, 19),
                      (19, 21), (19, 22), (20, 22), (24, 25), (25, 26), (26, 27), (27, 29), (27, 30), (28, 29)]

    neighbor_link = [(i, j) for (i, j) in neighbor_link_]
    edge = self_link + neighbor_link
    adjacency = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            for (x, y) in edge:
                if (x, y) == (i, j):
                    adjacency[i][j] = 1
                    adjacency[j][i] = adjacency[i][j]
    return adjacency


def get_xyz_flatten_adjacency(adjacency):
    num_node = adjacency.shape[0]
    A = np.zeros((num_node * 3, num_node * 3))
    for i in range(num_node):
        for j in range(num_node):
            A[i * 3][j * 3] = adjacency[i][j]
            A[i * 3][j * 3 + 1] = adjacency[i][j]
            A[i * 3][j * 3 + 2] = adjacency[i][j]
            A[i * 3 + 1][j * 3] = adjacency[i][j]
            A[i * 3 + 1][j * 3 + 1] = adjacency[i][j]
            A[i * 3 + 1][j * 3 + 2] = adjacency[i][j]
            A[i * 3 + 2][j * 3] = adjacency[i][j]
            A[i * 3 + 2][j * 3 + 1] = adjacency[i][j]
            A[i * 3 + 2][j * 3 + 2] = adjacency[i][j]
    return A
