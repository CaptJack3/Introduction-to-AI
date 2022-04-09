import numpy as np
from MazeProblem import MazeState, MazeProblem, compute_robot_direction
from Robot import UniformCostSearchRobot
from GraphSearch import NodesCollection


def tail_manhattan_heuristic(state: MazeState):
    # TODO (EX 7.2), implement heuristic, delete exception
    tail=state.tail
    goal=state.maze_problem.tail_goal
    manhattan_dist=abs(tail[0]-goal[0])+abs(tail[1]-goal[1])
    #operator_cost=abs(self.)
    return state.maze_problem.forward_cost*manhattan_dist
    
    



def center_manhattan_heuristic(state: MazeState):
    # TODO (EX 9.2), implement heuristic, delete exception
    # tail=state.tail
    # head=state.head
    goal_tail=state.maze_problem.tail_goal
    goal_head=state.maze_problem.head_goal
    robot_center=[(state.tail[0]+state.head[0])/2,(state.tail[1]+state.head[1])/2]
    goal_center=[(goal_tail[0]+goal_head[0])/2,(goal_tail[1]+goal_head[1])/2]
    manhattan_dist=abs(robot_center[0]-goal_center[0])+abs(robot_center[1]-goal_center[1])
    return state.maze_problem.forward_cost*manhattan_dist
    


class ShorterRobotHeuristic:
    def __init__(self, maze_problem: MazeProblem, k):
        assert k % 2 == 0, "odd must be even"
        assert maze_problem.length - k >= 3, f"it is not possible to shorten a {maze_problem.length}-length robot by " \
                                             f"{k} units because robot length has to at least 3"
        self.k = k
        ################################################################################################################
        # TODO (EX. 13.2): replace all three dots, delete exception
        # raise NotImplemented
        shorter_robot_head_goal, shorter_robot_tail_goal = self._compute_shorter_head_and_tails(maze_problem.head_goal,maze_problem.tail_goal)
        self.new_maze_problem = MazeProblem(maze_map=maze_problem.maze_map,
                                            initial_head=shorter_robot_tail_goal,
                                            initial_tail=shorter_robot_head_goal,
                                            head_goal=shorter_robot_head_goal,  # doesn't matter, don't change
                                            tail_goal=shorter_robot_tail_goal)  # doesn't matter, don't change
        self.node_dists = UniformCostSearchRobot().solve(self.new_maze_problem, compute_all_dists=True)
       
        
       # self.node_dists = UniformCostSearchRobot().solve(self.new_maze_problem, compute_all_dists=True)

        ################################################################################################################

        # assert isinstance(self.node_dists, NodesCollection)

    def _compute_shorter_head_and_tails(self, head, tail):
        # TODO (EX. 13.1): complete code here, delete exception
        # I need to make tuple ([head],[tail])
        # I need to be sure head and tail vectors are in numpy.ndarray class
        head_vec=np.array(head)
        tail_vec=np.array(tail) 
        d=compute_robot_direction(head_vec, tail_vec)
        new_head_vec=head_vec-d*((self.k)//2)
        new_tail_vec=tail_vec+d*((self.k)//2)
        return (new_head_vec,new_tail_vec)
        

    def __call__(self, state: MazeState):
        # TODO (EX. 13.3): replace each three dots, delete exception
        # raise NotImplemented
        shorter_head_location, shorter_tail_location = self._compute_shorter_head_and_tails(state.head,state.tail)
        new_state = MazeState(state.maze_problem, head=shorter_tail_location, tail=shorter_head_location)
        # mine stupied games
        
        #
        if new_state in self.node_dists:
            node = self.node_dists.get_node(new_state)
            return node.g_value
        else:
            # # mine stupied games
            # shorter_head_location, shorter_tail_location = self._compute_shorter_head_and_tails(state.maze_problem.head_goal,state.maze_problem.tail_goal)
            # new_state = MazeState(state.maze_problem, head=shorter_tail_location, tail=shorter_head_location)
            # if new_state in self.node_dists:
            #     node = self.node_dists.get_node(new_state)
            #     return node.g_value
            # else:
            # return state.maze_problem
            # return float("inf")
            return state.maze_problem.maze_map.size
            #
            # return tail_manhattan_heuristic(state) # maybe return the h(initial_node)
            # return 0 # what should we return in this case, so that the heuristic would be as informative as possible
                        # but still admissible

            # return float("inf")  # what should we return in this case, so that the heuristic would be as informative as possible
                        # but still admissible
