import copy
import math
import itertools
import time
from operator import itemgetter
from itertools import chain
from collections import Counter
from itertools import groupby
import random


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


def build_kdtree(points, depth=0):
    n = len(points)
    if n <= 0:
        return None
    axis = depth % 2
    if axis == 0:
        sorted_points = sorted(points, key=lambda point: point.x)
    else:
        sorted_points = sorted(points, key=lambda point: point.y)
    middle = n // 2
    return {
        'point': sorted_points[middle],
        'left': build_kdtree(sorted_points[:middle], depth=depth + 1),
        'right': build_kdtree(sorted_points[middle + 1:], depth=depth + 1)
    }


def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)

    if d1 < d2:
        if p1.value not in pivot.city:
            return p1
        return p2
    if p2.value not in pivot.city:
        return p2
    return p1


def kdtree_closest_point(root, point, depth=0):
    if root is None:
        return None

    axis = depth % 2

    point_axis = point.y
    root_axis = root['point'].y
    if axis == 0:
        point_axis = point.x
        root_axis = root['point'].x

    if point_axis < root_axis:
        next_branch = root['left']
        opposite_branch = root['right']
    else:
        next_branch = root['right']
        opposite_branch = root['left']

    best = closer_distance(point,
                           kdtree_closest_point(next_branch, point, depth + 1),
                           root['point']
                           )

    if distance(point, best) > abs(point_axis - root_axis):
        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch, point, depth + 1),
                               best
                               )
    return best


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


def closest_point(all_points, new_point, unreachable):
    best_point = None
    best_distance = float('inf')
    for current_point in all_points:
        if current_point.value == new_point.value:
            continue
        current_distance = distance(new_point, current_point)
        if current_distance < best_distance:
            if current_point.value in unreachable and not current_point.connected:
                best_distance = current_distance
                best_point = current_point

    return best_point, best_distance


class Node:
    def __init__(self, tup):
        self.value = tup
        self.x, self.y = tup.split(",")
        self.x = int(self.x)
        self.y = int(self.y)
        self.connected = False
        self.connectedNode = None
        self.connections = []
        self.connectedDistance = float('inf')
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
        sorted_cities = sorted(visitedNodes, key=lambda node: node)
        if sorted_cities not in city_list:
            city_list.append(sorted_cities)

        graph[i].city = sorted_cities

    return city_list, graph


def get_closest_points_graph(graph, root):
    closest_points = {}
    for i in graph:
        closest = kdtree_closest_point(root, graph[i])
        if closest:
            # print(i, closest.value, distance(closest, graph[i]))
            closest_points[i] = (closest.value, distance(closest, graph[i]))
    return closest_points


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



    edges = loadData()

    empty_set, string_edges = parse_data(edges)
    graph = createGraph(empty_set, string_edges)

    city_list, graph = get_cities_from_graph(graph)
    city_list = sorted(city_list)
    city_indices = [x for x in range(len(city_list))]
    # print(city_indices)
    routes = []
    union_find = uf_ds()
    union_find.make_set(empty_set)
    for i in city_list:
        for j in range(len(i) - 1):
            union_find.op_union(i[j], i[j + 1])
    # print(display(empty_set,union_find))
    root = build_kdtree(graph.values())

    new_edges = []

    for i in range(len(city_list)):
        for j in range(len(city_list[i])):
            point = city_list[i][j]
            closest_point = kdtree_closest_point(root, graph[point])
            distance_points = distance(closest_point, graph[point])
            new_edges.append((point, closest_point.value, distance_points))

    new_edges = sorted(new_edges, key=lambda x: x[2], reverse=False)
    print(new_edges)
    print(len(new_edges))
    cost = 0
    edge_counter = 0

    while edge_counter < len(city_list) and len(new_edges) > 0:
        u, v, w = new_edges.pop(0)
        if union_find.op_find(u) != union_find.op_find(v):
            union_find.op_union(u, v)
            edge_counter += 1
            cost += w
            routes.append((u, v))

    """for edge in new_edges:
        if union_find.op_find(edge[0]) != union_find.op_find(edge[1]):
            routes.append((edge[0], edge[1]))
            cost += edge[2]
            union_find.op_union(edge[0], edge[1])
    """
    print(sorted(routes))
    print(cost)
    print(all_equal(display(empty_set, union_find)))
    """while len(next_cities) > 0:
        for city in next_cities:
            city_distances = []
            print(len(city))
            for point in city:
                closest = kdtree_closest_point(root, graph[point])
                dist = distance(closest, graph[point])
                city_distances.append(((point, closest.value), dist))
            city_distances = sorted(city_distances, key=lambda x: x[1], reverse=False)
            #print(city_distances, city)

            min_distance = city_distances[0][1]
            for min_city in city_distances:

                if union_find.op_find(min_city[0][0]) != union_find.op_find(min_city[0][1]):
                    # print(union_find.op_find(min_city[0][0]), union_find.op_find(min_city[0][1]))

                    union_find.op_union(union_find.op_find(min_city[0][0]), union_find.op_find(min_city[0][1]))
                    next_city = graph[min_city[0][1]].city
                    this_city = graph[min_city[0][0]].city
                    graph[min_city[0][1]].city = next_city + this_city
                    if this_city in all_cities:
                        all_cities.remove(this_city)
                    if next_city in all_cities:
                        all_cities.remove(next_city)
                    next_cities.append(next_city)
                    routes.append(min_city[0])
            print(len(city_distances))
            next_cities.remove(city)

            union_array = display(empty_set, union_find)
            if len(next_cities) == 0 and not all_equal(union_array):
                print(len(all_cities))
                most_common = (Counter(union_array)).most_common()[0]
                random_element = all_cities[0]
                #print("random: ",random_element, most_common[0])
                # print(graph[random_element].city, union_array, random_element)
                next_cities.append(random_element)
            print(len(next_cities))
            print(len(routes) - len(city_list), " routes remaining")
            #print(city_distances, routes)
            #print()
    print("UNEIGFOEGW")
    """
    for route in routes:
        new_route = stringToTuple(route)
        edges.append(new_route)
    empty_set, string_edges = parse_data(edges)
    graph = createGraph(empty_set, string_edges)
    print(len(routes))
    total_distance = 0
    for route in routes:
        total_distance += distance_coordinates(route[0], route[1])
    print(total_distance)
    save_data(routes)


if __name__ == "__main__":
    main()
