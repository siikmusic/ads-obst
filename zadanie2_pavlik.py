import math
import time
from itertools import groupby
import random
from sklearn.neighbors import KDTree


def choice_excluding(lst, exception):
    possible_choices = [v for v in lst if v != exception]
    return random.choice(possible_choices)


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class uf_ds:
    parent_node = {}
    rank = {}

    def make_set(self, u):
        for i in u:
            self.parent_node[i] = i
            self.rank[i] = 0

    def op_find(self, k):
        if self.parent_node[k] == k:
            return k
        return self.op_find(self.parent_node[k])

    def op_union(self, a, b):
        x = self.op_find(a)
        y = self.op_find(b)
        self.parent_node[x] = y


def display(u, data):
    return [data.op_find(i) for i in u]

def distance_coordinates(point1, point2):
    point1, point2 = stringToTuple((point1, point2))
    x1, y1 = point1
    x2, y2 = point2

    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def distance(point1, point2):
    x1, y1 = point1.getTuple()
    x2, y2 = point2.getTuple()

    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


class Node:
    def __init__(self, tup):
        self.value = tup
        self.x, self.y = tup.split(",")
        self.x = int(self.x)
        self.y = int(self.y)
        self.connections = []
        self.city = None

    def print(self):
        return self.x, self.y

    def getTuple(self):
        return self.x, self.y


def createGraph(nodes, routes):
    graph = {}
    for node in nodes:
        node_object = Node(node)
        graph[node] = node_object
    for route in routes:
        node_from, node_to = route
        graph[node_from].connections.append(Node(node_to))
        graph[node_to].connections.append(Node(node_from))
    return graph


def nodeToString(node):
    return str(node[0]) + "," + str(node[1])


def stringToTuple(node_string_tuple):
    s1, s2 = node_string_tuple
    s1 = s1.split(",")
    s1[0] = int(s1[0])
    s1[1] = int(s1[1])
    s2 = s2.split(",")
    s2[0] = int(s2[0])
    s2[1] = int(s2[1])
    return (s1, s2)


def edgeToNodeString(edge):
    node1, node2 = edge
    node1 = nodeToString(node1)
    node2 = nodeToString(node2)
    return node1, node2


def getReachableNodes(graph, startingNode, visitedNodes):
    if startingNode in visitedNodes:
        return
    visitedNodes[startingNode] = True
    connections = graph[startingNode].connections
    for connection in connections:
        getReachableNodes(graph, connection.value, visitedNodes)


def getUnreachableNodes(graph, startingNode, nodes):
    visitedNodes = {}
    getReachableNodes(graph, startingNode, visitedNodes)
    return set(nodes) - set(visitedNodes.keys())


def loadData():
    edges = []
    with open('graf.txt', 'r') as f:
        for line in f:
            pair1, pair2 = line.strip()[1:-1].split('] [')

            x1, y1 = map(int, pair1.split(','))
            x2, y2 = map(int, pair2.split(','))

            edges.append(([x1, y1], [x2, y2]))
    return edges


def format_coordinate(c):
    return f"[{c}]"


def format_tuple(t):
    c1 = format_coordinate(t[0])
    c2 = format_coordinate(t[1])

    return f"{c1} {c2}"


def save_data(input_array):
    output_string = "\n".join(format_tuple(t) for t in input_array)

    with open("out.txt", "w") as f:
        f.write(output_string)


def parse_data(edges):
    empty_set = []
    string_edges = []
    for edge in edges:
        node1, node2 = edgeToNodeString(edge)
        if node1 not in empty_set:
            empty_set.append(node1)
        if node2 not in empty_set:
            empty_set.append(node2)

        string_edges.append((node1, node2))
    return empty_set, string_edges


def get_cities_from_graph(graph):
    city_list = []

    for i in graph:
        visitedNodes = {}
        getReachableNodes(graph, i, visitedNodes)
        sorted_cities = sorted(visitedNodes, key=lambda node: node, reverse=True)
        if sorted_cities not in city_list:
            city_list.append(sorted_cities)

        graph[i].city = sorted_cities

    return city_list, graph


def shortest_path(graph, city_list, union_find):
    new_edges = []
    routes = []
    start_time = time.time()
    for i in range(len(city_list)):
        edges_current =[graph[n] for n in city_list[i]]
        tree = KDTree([(edges_current[i].x, edges_current[i].y) for i in range(len(edges_current))])

        for j in range(i + 1, len(city_list)):

            for point in city_list[j]:
                distance, nearest = tree.query([[graph[point].x, graph[point].y]], k=1)
                distance = distance[0]
                nearest = edges_current[nearest[0][0]]
                new_edges.append((point, nearest.value, distance))

    end_time = time.time()
    print("All paths time: ", end_time - start_time)
    new_edges = sorted(new_edges, key=lambda x: x[2])
    cost = 0
    start_time = time.time()

    for edge in new_edges:
        if len(routes) == len(city_list) -1:
            break
        if union_find.op_find(edge[0]) != union_find.op_find(edge[1]):
            routes.append((edge[0], edge[1]))
            cost += edge[2]
            union_find.op_union(edge[0], edge[1])
    end_time = time.time()

    return cost, routes


def create_uf(vertices, city_list):
    union_find = uf_ds()
    union_find.make_set(vertices)
    for i in city_list:
        for j in range(len(i) - 1):
            union_find.op_union(i[j], i[j + 1])
    return union_find


def main():
    edges = [([1, 9], [3, 8]),
             ([3, 8], [3, 9]),
             ([4, 6], [4, 7]),
             ([6, 7], [6, 8]),
             ([7, 5], [7, 6])]

    edges = [([1, 3], [1, 4]),
             ([3, 4], [3, 3]),
             ([4, 3], [4, 2]),
             ([2, 6], [2, 7]),
             ([2, 7], [1, 7]),
             ([1, 9], [2, 9]),
             ([4, 2], [5, 2]),
             ([5, 2], [6, 2]),
             ([6, 2], [6, 3]),
             ([6, 3], [6, 4]),
             ([6, 4], [6, 5]),
             ([6, 5], [6, 6]),
             ([6, 6], [6, 7]),
             ([6, 7], [5, 7]),
             ([6, 7], [6, 8]),
             ([5, 7], [5, 6]),
             ([8, 2], [8, 3]),
             ([8, 6], [8, 7]),
             ([8, 8], [8, 9]),
             ([0, 6], [0, 7]),
             ([3, 7], [3, 8]),
             ]
    start_time = time.time()
    edges = loadData()

    vertices, string_edges = parse_data(edges)
    graph = createGraph(vertices, string_edges)

    city_list, graph = get_cities_from_graph(graph)

    union_find = create_uf(vertices, city_list)

    cost, routes = shortest_path(graph, city_list, union_find)
    end_time = time.time()
    print("Running time: ", start_time - end_time)
    print(sorted(routes))
    print(cost)
    print(all_equal(display(vertices, union_find)))

    for route in routes:
        new_route = stringToTuple(route)
        edges.append(new_route)
    vertices, string_edges = parse_data(edges)
    graph = createGraph(vertices, string_edges)
    unreachanble = getUnreachableNodes(graph, string_edges[0][0], vertices)
    print(len(unreachanble))
    print(len(routes))
    print(cost)
    save_data(routes)


if __name__ == "__main__":
    main()
