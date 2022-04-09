from heapdict import heapdict
from MazeProblem import MazeState


class Node:
    def __init__(self, state: MazeState, parent: "Node", g_value=None):
        assert isinstance(state, MazeState), "state should be of type MazeState"
        self.state = state
        self.parent = parent
        self.g_value = g_value

    def get_path(self):
        current_node = self
        path = [current_node]
        while current_node.parent is not None:
            path = [current_node.parent] + path  # adding parent to beginning of the path
            current_node = current_node.parent
        return path


class GraphSearchSolution:
    def __init__(self, final_node: Node, solve_time: float, n_node_expanded: int, init_heuristic_time=None,
                 no_solution_reason=None):
        if final_node is not None:
            self.cost = final_node.g_value
            self.path = final_node.get_path()
        else:
            assert no_solution_reason is not None
            self.no_solution_reason = no_solution_reason
            self.cost = float("inf")
            self.path = None
        self.solve_time = solve_time
        self.n_node_expanded = n_node_expanded
        self.init_heuristic_time = init_heuristic_time


class NodesCollection:
    def __init__(self):
        self._collection = dict()

    def add(self, node: Node):
        assert isinstance(node, Node)
        assert node.state not in self._collection
        self._collection[node.state] = node

    def remove_node(self, node):
        assert node.state in self._collection
        del self._collection[node.state]

    def __contains__(self, state):
        return state in self._collection

    def get_node(self, state):
        assert state in self._collection
        return self._collection[state]


class NodesPriorityQueue:
    def __init__(self):
        self.nodes_queue = heapdict()
        self.state_to_node = dict()

    def add(self, node, priority):
        assert node.state not in self.state_to_node
        self.nodes_queue[node] = priority
        self.state_to_node[node.state] = node

    def pop(self):
        if len(self.nodes_queue) > 0:
            node, priority = self.nodes_queue.popitem()
            del self.state_to_node[node.state]
            return node
        else:
            return None

    def __contains__(self, state):
        return state in self.state_to_node

    def get_node(self, state):
        assert state in self.state_to_node
        return self.state_to_node[state]

    def remove_node(self, node):
        assert node in self.nodes_queue
        del self.nodes_queue[node]
        assert node.state in self.state_to_node
        del self.state_to_node[node.state]

    def __len__(self):
        return len(self.nodes_queue)


class Queue:
    def __init__(self):
        self.queue = []

    def add(self, item):
        self.queue = [item] + self.queue

    def pop(self):
        if len(self.queue) == 0:
            return None
        item = self.queue[-1]
        del self.queue[-1]
        return item

    def __contains__(self, state):
        for node in self.queue:
            if node.state == state:
                return True
        return False

    def __len__(self):
        return len(self.queue)
