import pickle

def construct_vertices_edges():
    input_path = './data/corenet/cjkConcept.dat'
    f = open(input_path, 'r', encoding='utf-8')

    idx = 0

    vertices = set()
    edges = {}

    for line in f:
        idx += 1
        if (idx < 15):
            continue
        items = line.strip().split()

        korterm1 = items[1]
        korterm2 = items[2]

        vertices.add(korterm1)
        vertices.add(korterm2)

        if (korterm1 not in edges):
            edges[korterm1] = {}
        if (korterm2 not in edges):
            edges[korterm2] = {}

        edges[korterm1][korterm2] = 1
        edges[korterm2][korterm1] = 1

    f.close()
    return list(vertices),edges


def find_shortest_path(vertices, edges):
    INF = 5000
    idx = 0
    for k in vertices:
        idx += 1
        print (idx)
        for i in vertices:
            if (i==k):
                continue
            for j in vertices:
                if (i==j or j==k):
                    continue
                p_i_j = edges[i][j] if (i in edges and j in edges[i]) else INF
                p_i_k = edges[i][k] if (i in edges and k in edges[i]) else INF
                p_k_j = edges[k][j] if (k in edges and j in edges[k]) else INF

                if (p_i_k + p_k_j < p_i_j):
                    edges[i][j] = p_i_k + p_k_j
    return edges



def main():
    vertices, edges = construct_vertices_edges()
    edges = find_shortest_path(vertices, edges)
    pickle.dump(edges, open('./data/korterm_shortest_path.pickle', 'wb'))


if __name__ == '__main__':
    main()