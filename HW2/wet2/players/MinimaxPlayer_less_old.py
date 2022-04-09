"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
from SearchAlgos import MiniMax, State
import time
import numpy as np
from utils import debug_data_matching


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.player_id = -1
        self.my_pos = None
        self.rival_pos = None
        self.turn = int()
        self.use_regular_heuristic = bool()

        # time we still available of the whole game_time provided at the beginning
        self.time_limit_game_personal = game_time
        # time limit we set for each of our moves based on the time available for the whole game:
        # we assume less than 40 moves in a game for the initial value
        self.time_limit_move_personal = game_time / 100
        # time to leave at the end of the turn: do not start another computation of the DFS after this threshold
        self.extra_time = self.time_limit_move_personal
        self.time_array=[0,0,0,0,0,0,0]
        self.time_ndarray =[[0],[0],[0],[0],[0]]

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
        self.use_regular_heuristic = True
        self.SF0=0.3 # for moving the information to the game wrapper
        self.BF =40 # was 35
        self.SF1=4
        # raise NotImplementedError

    def make_move(self, time_limit) -> tuple:
        """
        DIRECTION = MOVE
        :param time_limit: float, time limit for a single turn.
        :return direction: tuple, specifying the Player's movement
        """

        # save starting time for turn
        # self.BF=35
        iter_start=0
        iter_finish=1
        iter_time=0
        # update personal time_limit for move based on how many turn have already been played
        # if self.turn > 0:
        #     self.time_limit_move_personal = self.time_limit_game_personal / (40 - self.turn)
        # else:
        #     self.time_limit_move_personal = self.time_limit_game_personal / 2
        # # self.vector.append(self.time_limit_move_personal)
        # # change extra_time parameter
        # self.time_limit_move_personal=np.inf
        # if self.turn < 18:
        #     self.extra_time = min(0.99 * time_limit, self.time_limit_move_personal)
        # else:
        #     self.extra_time = min(0.95 * time_limit, self.time_limit_move_personal)

        # # print for debug
        # print("Turn completed:", self.turn,
        #       "\t-\tMy player computing move for turn", self.turn + 1)

        # compute current state
        state = State(self.board, self.my_pos, self.rival_pos, self.use_regular_heuristic)
        # initialize search
        search_tree = MiniMax()
        # check if we have enough time; if so, call the search, increasing depth by 1 substitute the condition with
        # another one relating the time left in the turn and the branching factor
        depth = 1
        direction = None
        #while depth == 1 or time.time() - time_start < time_limit - self.extra_time:
        while depth==1 or time_end-time.time()>(iter_finish-iter_start)*self.BF+self.SF1:
        # while depth == 1 :
            # increase depth by 1 (iterative deepening approach)
            iter_time=time.time()
            depth += 1
            print("depth", depth)
            # if depth ==3:
            #     print(f"{time.time()-time_start},{time_limit},{self.extra_time}")
            #     print("stop")
            # search method returns a tuple: (minimax value, direction)
            # meas_time_init=time.time()
            direction = search_tree.search(state, depth=depth, turn=self.turn, maximizing_player=True)[1]
            # meas_time_finish = time.time()
            # self.time_array[depth]=max(self.time_array[depth],meas_time_finish-meas_time_init)
            # self.time_ndarray[depth].append(meas_time_finish-meas_time_init)
            # print(f"depth={depth},time_took={meas_time_finish-meas_time_init},turn={self.turn}")
            # print(f"turn={self.turn} ,time array={self.time_array}")
            # print(f"ndarray[depth],[turn]={self.time_ndarray}")
        # direction = search_tree.search(state, depth=depth, turn=self.turn, maximizing_player=True)[1]

        # if self.turn==35:
        #     print("pause")
        # if self.turn==10:
        #     print("pause")
        # if self.turn==25:
        #     print("pause")
        debug_data_matching(state.board,state.my_pos,state.rival_pos)

        # # print for debug
        # print("Minimax Move:", direction, "vector:", self.vector, "time:", time.time() - time_start)
        # print(f"turn={self.turn}")
        # print(f"depth2={self.time_ndarray[2]}")
        # print(f"depth3={self.time_ndarray[3]}")
        # print(f"depth4={self.time_ndarray[4]}")
        # if self.turn==19:
        #     print("stop")
        # update board with my move
        self.set_my_move(direction)
        # update time left in the game
        # self.time_limit_game_personal -= time.time() - time_start
        # iter_finish=time.time()
        return direction

    def set_rival_move(self, move: tuple):
        """
        Update your info, given the new position of the rival.
        :param move: tuple, the new position of the rival.
        :return: None
        """
        # check that the soldiers are set correctly TODO: delete afterwards
        all_soldiers = np.append(self.my_pos, self.rival_pos)
        for ind in range(24):
            if np.count_nonzero(all_soldiers == ind) is not [0, 1]:
                assert ValueError("two soldiers in the same position")

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

        if not debug_data_matching(self.board, self.my_pos, self.rival_pos):
            print("We got a problem here")

        # increase turn count
        self.turn += 1

    def set_my_move(self, move: tuple):
        """
                Update your info, given the new position of the rival.
                :param move: tuple, the new position of the rival.
                :return: None
                """
        if not debug_data_matching(self.board, self.my_pos, self.rival_pos):
            print("We got a problem here")
        if self.turn == 18:
            print("Minimax Player set my first move in stage 2")
        my_pos, my_soldier, rival_dead_pos = move

        # phase 1 of the game
        if self.turn < 18:
            self.board[my_pos] = 1
            self.my_pos[my_soldier] = my_pos
        # phase 2 of the game
        else:
            my_prev_pos = self.my_pos[my_soldier]
            self.board[my_prev_pos] = 0
            self.board[my_pos] = 1  # HERE WAS THE PROBLEM IT WAS: self.board[my_pos] = 2 INSTEAD OF =1
            self.my_pos[my_soldier] = my_pos

        # # print for debug
        # print("HELLO", move, rival_dead_pos)

        # if the rival ha achieved a mill in this turn
        if rival_dead_pos != -1:
            # # print for debug
            # print("Aiuto")
            # print(self.board[rival_dead_pos], self.board)

            self.board[rival_dead_pos] = 0

            # # print for debug
            # print(self.board[rival_dead_pos], self.board)

            dead_soldier = int(np.where(self.rival_pos == rival_dead_pos)[0][0])
            self.rival_pos[dead_soldier] = -2

        # increase turn count
        self.turn += 1

        if not debug_data_matching(self.board, self.my_pos, self.rival_pos):
            print("We got a problem here")
