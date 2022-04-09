"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
# TODO: you can import more modules, if needed
from SearchAlgos import MiniMax,AlphaBeta, State, Minimax_New
import time
import numpy as np
from utils import debug_data_matching


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        # TODO: property at ext line not used
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
        iter_finish = 1
        iter_time = 0
        time_end=time_start+time_limit-self.SF0
        # change extra_time parameter
        self.extra_time = 0.75 * time_limit

        # print for debug
        print("Turn completed:", self.turn,
              "\t-\tMy player computing move for turn", self.turn + 1)

        # compute current state
        # state = State(np.copy(self.board), np.copy(self.my_pos), np.copy(self.rival_pos),self.use_regular_heuristic)
        # initialize search
        # search_tree = MiniMax()
        search_tree=Minimax_New()
        # check if we have enough time; if so, call the search, increasing depth by 1 substitute the condition with
        # another one relating the time left in the turn and the branching factor
        depth = 1

        direction = None
        # TODO: find good condition on break
        # while time.time() - time_start < time_limit - self.extra_time:
        #     # increase depth by 1 (iterative deepening approach)
        #     depth += 1
        #     # search method returns a tuple: (minimax value, direction)
        #     max_value, direction = search_tree.search(state, depth=depth, turn=self.turn, maximizing_player=True)

        # # TODO: temporary (fixed depth)
        while depth==1 or time_end-time.time()>(iter_finish-iter_start)*self.BF+self.SF1:
            state = State(np.copy(self.board), np.copy(self.my_pos), np.copy(self.rival_pos),self.use_regular_heuristic)
            iter_start=time.time()
            debug_data_matching(state.board,state.my_pos,state.rival_pos)
            # ALPHA_VALUE_INIT = -np.inf
            # BETA_VALUE_INIT = np.inf  # !!!!!
            # alpha = ALPHA_VALUE_INIT
            # beta = BETA_VALUE_INIT
            direction = search_tree.search(state, depth, self.turn, True)[1]
            depth=depth+1
            iter_finish=time.time()

            # print for debug
            print("Minimax Move:", direction)

            # update board with my move
        self.set_my_move(direction)
        # update time left in the game
        #     self.time_limit_game_personal -= time.time() - time_start
        return direction

    def set_rival_move(self, move: tuple):
        """
        Update your info, given the new position of the rival.
        :param move: tuple, the new position of the rival.
        :return: None
        """
        debug_data_matching(self.board, self.my_pos, self.rival_pos)

        # check that the soldiers are set correctly TODO: delete afterwards
        # all_soldiers = np.append(self.my_pos, self.rival_pos)
        # for ind in range(24):
        #     if np.count_nonzero(all_soldiers == ind) is not [0, 1]:
        #         assert ValueError("two soldiers in the same position")

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


        if not debug_data_matching(self.board,self.my_pos,self.rival_pos):
            print("We got a problem here")

        # increase turn count
        self.turn += 1

    def set_my_move(self, move: tuple):
        """
                Update your info, given the new position of the rival.
                :param move: tuple, the new position of the rival.
                :return: None
                """
        if not debug_data_matching(self.board,self.my_pos,self.rival_pos):
            print("We got a problem here")
        if self.turn==18:
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
            self.board[my_pos] = 1   # HERE WAS THE PROBLEM IT WAS: self.board[my_pos] = 2 INSTEAD OF =1
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

            # # print for debug
            print("dead_soldier", dead_soldier, "\nmy_pos:", self.my_pos, "\nrival_pos:", self.rival_pos)
            print("move chosen:", move)

        # increase turn count
        self.turn += 1

        if not debug_data_matching(self.board,self.my_pos,self.rival_pos):
            print("We got a problem here")
