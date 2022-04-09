from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from MazeProblem import MazeProblem, MazeState
from Robot import Robot
from GraphSearch import GraphSearchSolution

global animation

class Animation:
    def __init__(self, solution: GraphSearchSolution, maze_problem: MazeProblem, robot: Robot, maze_name, blit=True):
        self.path = solution.path
        self.maze_map = maze_problem.maze_map
        self.maze_problem = maze_problem
        self.frame_index = 0
        self.head_goal = maze_problem.head_goal
        self.tail_goal = maze_problem.tail_goal
        self.maze_height = len(self.maze_map)
        self.maze_width = len(self.maze_map[0])
        self.blit = blit

        aspect = len(self.maze_map[0]) / len(self.maze_map)

        self.board_colors = {0: 'white', -1: 'gray', "0": 'white', "-1": 'gray'}
        # robot body is colored at each frame
        self.fig = plt.figure(frameon=False, figsize=(8 * aspect, 8))
        title = f"{maze_name}\n" \
                f"{robot.name}\n" \
                f"cost: {solution.cost}, solve time: {round(solution.solve_time, 2)}\n" \
                f"n nodes expanded: {solution.n_node_expanded}\n"
        self.fig.suptitle(title)
        self.ax = plt.axes(xlim=(0, len(self.maze_map[0])), ylim=(0, len(self.maze_map)))
        # self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)

        self.map_patches = [[None for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        for i in range(len(self.maze_map)):
            for j in range(len(self.maze_map[0])):
                face_color = self.board_colors[self.maze_map[i, j]]
                if np.all((i, j) == self.maze_problem.head_goal):
                    face_color = (0, 0.6, 0)
                elif np.all((i, j) == self.maze_problem.tail_goal):
                    face_color = (1, 0.9, 0)
                r = Rectangle((j, i), 1, 1, facecolor=face_color, edgecolor='black', fill=True)
                self.ax.add_patch(r)
                self.map_patches[i][j] = r

        # create boundary patch
        x_min = -0.5
        y_min = -0.5
        x_max = len(self.maze_map[0]) + 0.5
        y_max = len(self.maze_map) + 0.5
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        global animation
        animation = FuncAnimation(self.fig, func=self._animation_func, frames=len(self.path), interval=200, blit=self.blit,
                                  init_func=self._init_animation)
        self.animation = animation
        self.prev_patches_updated = []

    def get_animation(self):
        return self.animation

    def _init_animation(self):
        return [c for r in self.map_patches for c in r]

    def create_board_patches(self, state: MazeState):
        if not self.blit:
            for r in self.prev_patches_updated:
                r.set_facecolor("white")

        self.prev_patches_updated = []
        l = self.maze_problem.length
        for k, (i, j) in enumerate(self.maze_problem._iterate_on_robot_locations(state)):
            r = self.map_patches[i][j]
            r.set_facecolor(((1-k/l), (1-k/l) + k/l, 0))
            self.prev_patches_updated.append(r)
        if not self.blit:
            return [c for r in self.map_patches for c in r]
        else:
            return self.prev_patches_updated

    def _animation_func(self, i):
        return self.create_board_patches(self.path[i].state)

    def save(self, file_name, speed):
        self.animation.save(
            file_name,
            fps=10 * speed,
            dpi=200,
            savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

    @staticmethod
    def show():
        plt.show()

    # def init_func(self):
    #     for p in self.board_patch + sum(self.map_patches, []) + self.agent_patches:
    #         self.ax.add_patch(p)
    #     return self.board_patch + sum(self.map_patches, []) + self.agents
