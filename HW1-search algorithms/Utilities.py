from MazeProblem import create_problem
from Animation import Animation

def test_robot(robot: type, maps_indices, **robot_params):
    assert isinstance(robot, type), "param robot should be the class itself, not a class instance"
    solutions = []
    for i in maps_indices:
        robot_instance = robot(**robot_params)
        maze_problem = create_problem(f"maze_{i}")
        solution = robot_instance.solve(maze_problem)
        print(f"{robot_instance.name} solved maze_{i} in {round(solution.solve_time, 2)} seconds. "
              f"solution cost = {solution.cost}, "
              f"expanded {solution.n_node_expanded} nodes.")
        solutions.append(solution)
    return solutions


def solve_and_display(robot: type, maze_index, blit=True, **robot_params):
    assert isinstance(robot, type)
    maze_file = f"maze_{maze_index}"
    maze_problem = create_problem(maze_file)
    robot_instance = robot(**robot_params)
    solution = robot_instance.solve(maze_problem)
    print(f"{robot_instance.name} solved {maze_file} in {round(solution.solve_time, 2)} seconds. "
          f"solution costs = {solution.cost}, "
          f"# node expanded = {solution.n_node_expanded}.")
    if solution.path is not None:
        a = Animation(solution, maze_problem, robot_instance, maze_file, blit=blit)
        a.show()
        return a
    else:
        print(f"solution not found because {solution.no_solution_reason}")
