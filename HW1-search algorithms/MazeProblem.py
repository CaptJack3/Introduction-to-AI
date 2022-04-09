import numpy as np
import pandas as pd

def compute_robot_direction(head, tail):
    direction = np.sign(head - tail)
    assert np.sum(np.abs(direction)) == 1, f"invalid robot location. head: {head}, tail: {tail}"
    return direction

class MazeState:
    def __init__(self, maze_problem: "MazeProblem", head, tail):
        assert isinstance(head, np.ndarray), "head must be of type ndarray (numpy array)"
        assert isinstance(tail, np.ndarray), "tail must be of type ndarray (numpy array)"
        self.maze_problem = maze_problem
        self.tail = tail
        self.head = head
        self.hash_array = np.concatenate([self.tail, self.head])
        assert len(np.argwhere((self.head - self.tail) == 0)) == 1
        assert len(np.argwhere((self.maze_problem.head_goal - self.maze_problem.tail_goal) == 0)) == 1

    def robot_direction(self):
        return compute_robot_direction(self.head, self.tail)

    def __hash__(self):
        return hash(str(self.hash_array))

    def __eq__(self, other: "MazeState"):
        return np.all(self.hash_array == other.hash_array)

class MazeProblem:
    def __init__(self, maze_map: np.ndarray, initial_head: np.ndarray, initial_tail: np.ndarray, head_goal: np.ndarray,
                 tail_goal: np.ndarray, forward_cost=1, turn_cost=5):
        assert all([isinstance(x, np.ndarray) for x in [maze_map, initial_head, initial_tail, head_goal, tail_goal]])

        self.tail_goal = tail_goal
        self.head_goal = head_goal
        self.maze_map = maze_map
        self.initial_state = MazeState(self, initial_head, initial_tail)
        self.length = np.sum(np.abs(self.initial_state.head - self.initial_state.tail)) + 1
        assert self.length % 2 == 1, f"the robot length must be an odd number, got length {self.length}"
        assert np.sum(np.abs(head_goal - tail_goal)) + 1 == self.length, "invalid initial and goal locations"
        self._check_body_not_on_wall()
        self.forward_cost = forward_cost
        self.turn_cost = turn_cost

        self.right_rotate_dict = {
            (1, 0): np.array([-1, -1]),
            (0, 1): np.array([1, -1]),
            (0, -1): np.array([-1, 1]),
            (-1, 0): np.array([1, 1])
        }

        self.left_rotate_dict = {
            (1, 0): np.array([-1, 1]),
            (0, 1): np.array([-1, -1]),
            (0, -1): np.array([1, 1]),
            (-1, 0): np.array([1, -1])
        }

    def _iterate_on_robot_locations(self, state):
        # this iterates from tail to head
        robot_direction = state.robot_direction()
        assert np.sum(np.abs(robot_direction)) > 0, (state.head, state.tail)
        axis = np.argwhere(robot_direction != 0).reshape(1)[0]

        axis_direction = 1 if state.tail[axis] < state.head[axis] else -1
        head_plus_1 = 1 if state.tail[axis] < state.head[axis] else -1

        for i in range(state.tail[axis], state.head[axis] + head_plus_1, axis_direction):
            loc = state.head.copy()  # could take tail as well
            loc[axis] = i
            assert np.all(robot_direction == state.robot_direction())
            yield loc

    def _check_body_not_on_wall(self):
        for loc in self._iterate_on_robot_locations(self.initial_state):
            assert self.maze_map[loc[0], loc[1]] != -1, "invalid maze, the robot is located on a wall"

    def _check_turn_area_free_(self, prev_location: np.array, new_location: np.array):
        for i in range(min(prev_location[0], new_location[0]), max(prev_location[0], new_location[0]) + 1):
            for j in range(min(prev_location[1], new_location[1]), max(prev_location[1], new_location[1]) + 1):
                if not self._location_in_bounds_(np.array([i, j])):
                    return False
                if self._is_wall(np.array([i, j])):
                    return False
        return True

    def _location_in_bounds_(self, loc):
        return np.all(np.zeros(2) <= loc) and np.all(loc < self.maze_map.shape)

    def _is_wall(self, loc):
        i, j = loc
        return self.maze_map[i, j] == -1

    def _get_head_and_tail_location_for_turn(self, state: MazeState, turn_direction):
        if turn_direction == "right":
            rotate_dict = self.right_rotate_dict
        else:
            rotate_dict = self.left_rotate_dict
        robot_direction = state.robot_direction()
        new_head_location = (state.head + rotate_dict[tuple(robot_direction)] * (self.length - 1) / 2).astype("int")
        new_tail_location = (state.tail - rotate_dict[tuple(robot_direction)] * (self.length - 1) / 2).astype("int")
        assert len(np.argwhere((new_head_location - new_tail_location) == 0)) == 1
        return new_head_location, new_tail_location

    def _get_new_location_for_forward(self, state: MazeState):
        robot_direction = state.robot_direction()
        return state.head + robot_direction, state.tail + robot_direction

    def _create_state(self, head, tail):
        return MazeState(self, head=head, tail=tail)

    def expand_state(self, state):
        # forward move:
        new_head_location, new_tail_location = self._get_new_location_for_forward(state)

        if self._location_in_bounds_(new_head_location) and not self._is_wall(new_head_location):
            assert self._location_in_bounds_(new_tail_location) and not self._is_wall(new_tail_location)
            new_state = self._create_state(new_head_location, new_tail_location)
            yield new_state, self.forward_cost

        # right turn:
        new_head_location, new_tail_location = self._get_head_and_tail_location_for_turn(state, "right")
        if self._check_turn_area_free_(state.head, new_head_location) and \
                self._check_turn_area_free_(state.tail, new_tail_location):
            new_state = self._create_state(new_head_location, new_tail_location)
            yield new_state, self.turn_cost

        # left turn:
        new_head_location, new_tail_location = self._get_head_and_tail_location_for_turn(state, "left")
        if self._check_turn_area_free_(state.head, new_head_location) and \
                self._check_turn_area_free_(state.tail, new_tail_location):
            new_state = self._create_state(new_head_location, new_tail_location)
            yield new_state, self.turn_cost

    def is_goal(self, state: MazeState):
        return np.all(state.head == self.head_goal) and np.all(state.tail == self.tail_goal)

def create_problem(maze_name, forward_cost=1, turn_cost=5):
    maze_map = pd.read_csv(f"Mazes/{maze_name}.csv", header=None).to_numpy()
    assert np.sum(maze_map == 1) == 1, np.sum(maze_map == 1)
    assert np.sum(maze_map == 2) == 1, np.sum(maze_map == 2)
    assert np.sum(maze_map == 3) == 1, np.sum(maze_map == 3)
    assert np.sum(maze_map == 4) == 1, np.sum(maze_map == 4)
    tail = np.argwhere(maze_map == 1)[0]
    head = np.argwhere(maze_map == 2)[0]
    tail_goal = np.argwhere(maze_map == 3)[0]
    head_goal = np.argwhere(maze_map == 4)[0]
    maze_map[tail[0], tail[1]] = 0
    maze_map[head[0], head[1]] = 0
    maze_map[tail_goal[0], tail_goal[1]] = 0
    maze_map[head_goal[0], head_goal[1]] = 0
    maze_problem = MazeProblem(maze_map, head, tail, head_goal, tail_goal, forward_cost, turn_cost)
    return maze_problem

