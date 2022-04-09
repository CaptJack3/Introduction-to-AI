"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
from SearchAlgos import Minimax, State
import time
import numpy as np
from utils import debug_data_matching


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.game_time=game_time
        self.player_id = -1
        self.my_pos = None
        self.rival_pos = None
        self.turn = int()
        self.extra_time = 0
        self.use_regular_heuristic= bool()


    def set_game_params(self, board):
        """
        Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        :param board: numpy.array, of the board.
        :return: None
        """
        self.board = board
        self.my_pos = np.full(9, -1)
        self.rival_pos = np.full(9, -1)
        self.turn = 0
        self.use_regular_heuristic=True
        self.SF0=0.3
        self.SF1=3
        self.BF=35
        # raise NotImplementedError

    def make_move(self, time_limit) -> tuple:
        """
        DIRECTION = MOVE
        :param time_limit: float, time limit for a single turn.
        :return direction: tuple, specifying the Player's movement
        """
        time_start = time.time()
        iter_start = 0
        iter_finish = 0
        iter_time = 0
        time_end=time_start+time_limit-self.SF0

        # state = State(np.copy(self.board), np.copy(self.my_pos), np.copy(self.rival_pos),self.use_regular_heuristic)
        # search_tree = MiniMax()
        search_tree=Minimax()
        depth = 1
        direction = None
        if self.turn<35:
            while depth==1 or time_end-time.time()>(iter_finish-iter_start)*self.BF+self.SF1:
                # print(f"depth={depth}")
                state = State(np.copy(self.board), np.copy(self.my_pos), np.copy(self.rival_pos),self.use_regular_heuristic)
                iter_start=time.time()
                direction = search_tree.search(state, depth, self.turn, True)[1]
                depth=depth+1
                iter_finish=time.time()
            self.set_my_move(direction)
            print(f"turn={self.turn}")
        else:
            while depth<=3 and time_end-time.time()>(iter_finish-iter_start)*self.BF+self.SF1:
                # print(f"depth={depth}")
                state = State(np.copy(self.board), np.copy(self.my_pos), np.copy(self.rival_pos),self.use_regular_heuristic)
                iter_start=time.time()
                direction = search_tree.search(state, depth, self.turn, True)[1]
                depth=depth+1
                iter_finish=time.time()
            self.set_my_move(direction)
            # print(f"turn={self.turn}")
        return direction

    def set_rival_move(self, move: tuple):
        """
        Update your info, given the new position of the rival.
        :param move: tuple, the new position of the rival.
        :return: None
        """
        rival_pos, rival_soldier, my_dead_pos = move
        # phase 1 of the game
        if self.turn < 18:
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos
        # phase 2 of the game
        else:
            rival_prev_pos = self.rival_pos[rival_soldier]
            self.board[rival_prev_pos] = 0
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos
        # if the rival ha achieved a mill in this turn
        if my_dead_pos != -1:
            self.board[my_dead_pos] = 0
            dead_soldier = int(np.where(self.my_pos == my_dead_pos)[0][0])
            self.my_pos[dead_soldier] = -2

        # increase turn count
        self.turn += 1

    def set_my_move(self, move: tuple):
        """
                Update your info, given the new position of the rival.
                :param move: tuple, the new position of the rival.
                :return: None
                """
        my_pos, my_soldier, rival_dead_pos = move

        # phase 1 of the game
        if self.turn < 18:
            self.board[my_pos] = 1
            self.my_pos[my_soldier] = my_pos
        # phase 2 of the game
        else:
            my_prev_pos = self.my_pos[my_soldier]
            self.board[my_prev_pos] = 0
            self.board[my_pos] = 1   # HERE WAS THE PROBLEM IT WAS: self.board[my_pos] = 2 INSTEAD OF =1
            self.my_pos[my_soldier] = my_pos
        # if the rival ha achieved a mill in this turn
        if rival_dead_pos != -1:
            self.board[rival_dead_pos] = 0
            dead_soldier = int(np.where(self.rival_pos == rival_dead_pos)[0][0])
            self.rival_pos[dead_soldier] = -2

            # # # print for debug
            # print("dead_soldier", dead_soldier, "\nmy_pos:", self.my_pos, "\nrival_pos:", self.rival_pos)
            # print("move chosen:", move)

        # increase turn count
        self.turn += 1

        # if not debug_data_matching(self.board,self.my_pos,self.rival_pos):
        #     print("We got a problem here")
