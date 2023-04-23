import copy
import math
import itertools
from operator import itemgetter


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
             ([6, 7], [5, 7]),
             ([5, 7], [5, 6]),
             ([8, 2], [8, 3]),
             ([8, 6], [8, 7]),
             ([8, 8], [8, 9]),
             ([0, 6], [0, 7]),
             ([3, 7], [3, 8]),
             ]

    # edges = loadData()
    empty_set, string_edges = parse_data(edges)
    graph = createGraph(empty_set, string_edges)
    graph_copy = graph.copy()
    city_list, graph = get_cities_from_graph(graph)
    city_list = sorted(city_list)
    root = build_kdtree(graph.values())

    routes = []

    print(city_list)
    print(graph.values())
    connected_cities = {}
    for i in range(len(city_list)):
        connected_cities[i] = False
    for i in range(len(city_list) - 1):
        current_city = []
        root = build_kdtree(graph.values())
        for j in range(len(city_list[i])):
            point = city_list[i][j]
            closest_point = kdtree_closest_point(root, graph[point])
            if closest_point is None:
                print(graph[point].value)
                del graph[point]
                continue

            distance_points = distance(closest_point, graph[point])
            # print((point, closest_point.value), distance_points)
            current_city.append(((point, closest_point.value), distance_points))
            print(graph[point].value)
            del graph[point]

        current_city = sorted(current_city, key=lambda x: x[1], reverse=False)
        print(current_city)
        cheapest = current_city[0][1]
        for route in current_city:
            if route[1] == cheapest:
                routes.append(route[0])

    for route in routes:
        new_route = stringToTuple(route)
        edges.append(new_route)
    print(edges)
    empty_set, string_edges = parse_data(edges)
    graph = createGraph(empty_set, string_edges)
    unreachanble = getUnreachableNodes(graph, string_edges[0][0], empty_set)
    print(len(routes))
    print(routes)
    print(unreachanble)
    save_data(routes)


if __name__ == "__main__":
    main()
