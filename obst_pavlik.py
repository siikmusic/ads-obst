INT_MAX = 2147483647


class BSTNode:
    """
        Class for tree node
        visualization source: https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
    """

    def __init__(self, val=None, string=None):
        self.left = None
        self.right = None
        self.val = val
        self.string = string

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.string
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.string
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.string
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.string
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


def print_matrix(matrix):
    """Utility function to print matrix with rows on new line"""
    for i in matrix:
        print(i, "\n")


def find_string_index_row_2d(arr, string):
    """
    Utility function to find index of string in 2D array of tuples
    :param arr: 2D array of tuples
    :param string: searched string
    :return:
    """
    i = 0
    for row in arr:
        if row[1] == string:
            return i
        i += 1


def get_sorted_inputs():
    """
    Function that loads data from file, computes the frequency of words in dictionary and for getting keys for OBST
    :return: dictionary frequency - frequency of words in dictionary, clean_data - 2D tuple of pairs [frequency, word],
    keys - keys used for constructing OBST
    """
    dictionary_frequency = 0
    clean_data = []

    for f in open("dictionary.txt"):
        term_frequency = int(f.split(" ")[0])
        term = f.split(" ")[1].split("\n")[0]
        dictionary_frequency = dictionary_frequency + term_frequency
        clean_data.append([term_frequency, term])
    clean_data.sort(key=lambda x: x[1])

    keys = []

    for line in clean_data:
        if line[0] > 50000:
            keys.append(line[1])

    return dictionary_frequency, clean_data, keys


def calculate_p(clean_data, dictionary_frequency):
    """
    Function that calculates the probability of key being searched for
    :param clean_data: 2D array of tuples representing words and their frequency
    :param dictionary_frequency: summed frequency of words in dictionary
    :return: array of probabilities p
    """
    p = []
    for line in clean_data:
        if line[0] > 50000:
            p.append(line[0] / dictionary_frequency)
    return p


def calculate_q(clean_data, dictionary_frequency, keys):
    """
    Function to calculate the probability q of words being between 2 keys.
    :param clean_data: 2D array of tuples representing words and their frequency
    :param dictionary_frequency: summed frequency of words in dictionary
    :param keys: keys for the optimal BST
    :return: array of probabilities q
    """
    q = []
    index_from = 0
    index_to = find_string_index_row_2d(clean_data, keys[0]) - 1
    frequency_between = 0

    # calculate mean frequency of words from beginning of dictionary to the first key
    for j in range(index_from, index_to):
        frequency_between += clean_data[j][0]
    q.append(frequency_between / dictionary_frequency)
    # calculate mean frequency of words first key to rest and also from last to the end of dictionary
    for i in range(len(keys)):
        k_i = keys[i]

        if i != len(keys) - 1:
            k_i_1 = keys[i + 1]
            index_to = find_string_index_row_2d(clean_data, k_i_1)

        else:
            index_to = len(clean_data)

        index_from = find_string_index_row_2d(clean_data, k_i) + 1
        frequency_between = 0
        for j in range(index_from, index_to):
            frequency_between += clean_data[j][0]
        q.append(frequency_between / dictionary_frequency)
    return q


def get_data():
    dictionary_frequency, clean_data, keys = get_sorted_inputs()

    p = calculate_p(clean_data, dictionary_frequency)

    q = calculate_q(clean_data, dictionary_frequency, keys)

    return p, q, clean_data, keys


def OBST(p, q, n):
    """
    Function for optimal binary search tree construction
    :param p:
    :param q:
    :param n:
    :return: root - tableaux with keys in OBST
    """
    e = [[None for _ in range(n + 1)] for _ in range(n + 2)]
    w = [[None for _ in range(n + 1)] for _ in range(n + 2)]
    root = [[None for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]

    for l in range(1, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            e[i][j] = INT_MAX
            w[i][j] = w[i][j - 1] + p[j - 1] + q[j]
            for r in range(i, j + 1):
                t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if t < e[i][j]:
                    e[i][j] = t
                    root[i][j] = r
    return e, root


def constuct_OBST(root, n, keys):
    """
    Function that constructs OBST
    :param root: tableaux used to create optimal binary search tree
    :param n: number of keys
    :param keys: array of keys
    :return: root node of tree
    """
    r = BSTNode(root[1][n], keys[root[1][n] - 1])
    s = [[r, 1, n]]
    while len(s) > 0:
        u, i, j = s.pop()
        l = root[i][j]
        if l < j:
            v = BSTNode(root[l + 1][j], keys[root[l + 1][j] - 1])
            u.right = v
            s.append([v, l + 1, j])
        if i < l:
            v = BSTNode(root[i][l - 1], keys[root[i][l - 1] - 1])
            u.left = v
            s.append([v, i, l - 1])
    return r


def pocet_porovnani(in_string: str, root, keys, searched=0):
    """
    Function for number of comparisons needed to make in order to find string.
    Modified implementation of: https://www.geeksforgeeks.org/binary-search-tree-set-1-search-and-insertion/ to make it
    work for string comparisons
    :param in_string: searched string
    :param root: root of OBST
    :param keys: array of keys
    :param searched: number of times needed to search
    :return: returns number of times string was needed to search in bsd
    """
    if root is not None:
        searched += 1

    # Base Cases: root is null or string is present at root
    if root is None or root.string == in_string:
        return searched

    # Searched string is greater than root's string
    if root.string < in_string:
        return pocet_porovnani(in_string, root.right, keys, searched)

    # Searched string is smaller than root's string
    return pocet_porovnani(in_string, root.left, keys, searched)


if __name__ == "__main__":
    p_table, q_table, sorted_data, tree_keys = get_data()
    e, root = OBST(p_table, q_table, len(tree_keys))
    root_node = constuct_OBST(root, len(tree_keys), tree_keys)
    root_node.display()

    print(pocet_porovnani("must", root_node, tree_keys))
