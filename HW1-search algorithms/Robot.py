from MazeProblem import MazeProblem
from abc import ABC, abstractmethod
from GraphSearch import Node, NodesCollection, NodesPriorityQueue, GraphSearchSolution, Queue
from time import process_time as curr_time


class Robot(ABC):
    def __init__(self):
        self.name = "abstract robot"
        ...

    @abstractmethod
    def solve(self, maze: MazeProblem):
        ...


class BreadthFirstSearchRobot(Robot):
    def __init__(self):
        super(BreadthFirstSearchRobot, self).__init__()
        self.queue = None
        self.close = None
        self.name = "breadth first search robot"

    def solve(self, maze: MazeProblem, time_limit=float("inf")):
        start_time = curr_time()

        self.queue = Queue()
        self.close = NodesCollection()

        initial_node = Node(maze.initial_state, None, g_value=0)
        self.queue.add(initial_node)

        n_node_expanded = 0  # count the number of nodes expanded during the algorithm run.

        while True:
            if curr_time() - start_time >= time_limit:
                no_solution_found = True
                no_solution_reason = "time limit exceeded"
                break

            next_node = self.queue.pop()
            if next_node is None:
                no_solution_found = True
                no_solution_reason = "no solution exists"
                break

            self.close.add(next_node)
            ############################################################################################################
            # TODO (EX. 4.1): complete code here, delete exception
            n_node_expanded=n_node_expanded+1
            for s in next_node.state.maze_problem.expand_state(next_node.state):
                zz = Node(s[0], next_node, s[1] + next_node.g_value)
                if zz.state not in self.queue:
                    if zz.state.maze_problem.is_goal(zz.state):
                        # return zz.get_path()
                        return GraphSearchSolution(zz, curr_time() - start_time, n_node_expanded)
                    # if zz.state not in self.close._collection:
                    if zz.state not in self.close:
                        self.queue.add(zz)
                    # if zz.state in self.close:
                        # counter = counter + 1
            # raise NotImplemented

            ############################################################################################################
        # If we are here, then we didn't find a solution during the search
        assert no_solution_found
        return GraphSearchSolution(final_node=None, solve_time=curr_time() - start_time,
                                   n_node_expanded=n_node_expanded, no_solution_reason=no_solution_reason)


class BestFirstSearchRobot(Robot):
    def __init__(self):
        super(BestFirstSearchRobot, self).__init__()
        self.open = None
        self.close = None
        self.name = "abstract best first search robot"

    @abstractmethod
    def _calc_node_priority(self, node):
        ...

    def solve(self, maze_problem: MazeProblem, time_limit=float("inf"), compute_all_dists=False):
        start_time = curr_time()

        self.open = NodesPriorityQueue()
        self.close = NodesCollection()

        if hasattr(self, "_init_heuristic"):  # some heuristics need to be initialized with the maze problem
            init_heuristic_start_time = curr_time()
            self._init_heuristic(maze_problem)
            init_heuristic_time = curr_time() - init_heuristic_start_time
        else:
            init_heuristic_time = None

        initial_node = Node(maze_problem.initial_state, None, g_value=0)
        initial_node_priority = self._calc_node_priority(initial_node)
        self.open.add(initial_node, initial_node_priority)

        n_node_expanded = 0  # count the number of nodes expanded during the algorithm run.

        while True:
            if curr_time() - start_time >= time_limit:
                no_solution_found = True
                no_solution_reason = "time limit exceeded"
                break

            next_node = self.open.pop()
            if next_node is None:
                no_solution_found = True
                no_solution_reason = "no solution exists"
                break

            self.close.add(next_node)
            if maze_problem.is_goal(next_node.state):
                if not compute_all_dists:  # we will use this later, don't change
                    return GraphSearchSolution(next_node, solve_time=curr_time() - start_time,
                                               n_node_expanded=n_node_expanded, init_heuristic_time=init_heuristic_time)
            ############################################################################################################
            # TODO (EX. 5.1): complete code here, delete exception
            n_node_expanded=n_node_expanded+1
            
            for s in next_node.state.maze_problem.expand_state(next_node.state):
                new_cost=next_node.g_value+s[1]
                if s[0] not in self.close:
                   # new_cost=next_node.g_value+s[1]
                   # check if same s is already in Open
                   if s[0] in self.open:    
                       old_node=self.open.get_node(s[0]) # this is node
                       if old_node.g_value>new_cost:
                           #starts here
                           self.open.remove_node(old_node)
                           zz=Node(s[0],next_node,new_cost)
                           priori=new_cost
                           priori=self._calc_node_priority(zz)
                           self.open.add(zz, priori)
                   else:
                       zz=Node(s[0],next_node,new_cost)
                       priori=new_cost
                       priori=self._calc_node_priority(zz)
                       self.open.add(zz, priori)
                else: # state s[0] in close
                    n_curr=self.close.get_node(s[0])
                    if new_cost<n_curr.g_value:
                        self.close.remove_node(n_curr)
                        zz=Node(s[0],next_node,new_cost)
                        priori=self._calc_node_priority(zz)
                        self.open.add(zz,priori)
                       
                                       
                           
                          
            #raise NotImplemented

            ############################################################################################################

        if compute_all_dists:
            return self.close
        else:
            assert no_solution_found
            return GraphSearchSolution(final_node=None, solve_time=curr_time() - start_time,
                                       n_node_expanded=n_node_expanded, no_solution_reason=no_solution_reason,
                                       init_heuristic_time=init_heuristic_time)


class UniformCostSearchRobot(BestFirstSearchRobot):
    def __init__(self):
        super(UniformCostSearchRobot, self).__init__()
        self.name = "uniform cost search robot"

    def _calc_node_priority(self, node):
        # TODO (Ex. 5.2): complete code here (just return the g value), delete exception
        return node.g_value
        raise NotImplemented

class WAStartRobot(BestFirstSearchRobot):
    def __init__(self, heuristic, w=0.5, **h_params):
        super().__init__()
        assert 0 <= w <= 1
        self.heuristic = heuristic
        self.orig_heuristic = heuristic  # in case heuristic is an object function, we need to keep the class
        self.w = w
        self.name = f"wA* [{self.w}, {heuristic.__name__}]" if len(h_params) == 0 else \
            f"wA* [{self.w}, {heuristic.__name__}, {h_params}]"
        self.h_params = h_params

    def _init_heuristic(self, maze_problem):
        if isinstance(self.orig_heuristic, type):
            self.heuristic = self.orig_heuristic(maze_problem, **self.h_params)

    def _calc_node_priority(self, node):
        # TODO (Ex. 7.1): complete code here, delete exception
        huristic_val=self.heuristic(node.state)
        # if node.parent==None:  # thos is equal to check node.g_valie isn't 0
        #     operation_cost=node.g_value
        # else:
        #     operation_cost=node.g_value-node.parent.g_value
        return (1-self.w)*node.g_value+self.w*huristic_val
        raise NotImplemented
