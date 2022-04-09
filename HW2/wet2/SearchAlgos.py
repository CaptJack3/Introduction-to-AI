"""Search Algos: MiniMax, AlphaBeta
"""
from __future__ import annotations
import numpy as np
from utils import debug_data_matching, get_directions

ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf  # !!!!!


class SearchAlgos:
    # def __init__(self, utility, succ, perform_move=None, goal=None):
    #     """The constructor for all the search algos.
    #     You can code these functions as you like to,
    #     and use them in MiniMax and AlphaBeta algos as learned in class
    #     :param utility: The utility function.
    #     :param succ: The successor function.
    #     :param perform_move: The perform move function.
    #     :param goal: function that check if you are in a goal state.
    #     """
    #     self.utility = utility
    #     self.succ = succ
    #     self.perform_move = perform_move
    #     self.goal = goal

    def __init__(self):
        pass

    def search(self, state: State, depth: int, turn: int, maximizing_player):
        pass


# class MiniMax_Old(SearchAlgos):
#     def __init__(self):
#         SearchAlgos.__init__(self)
#         pass
#
#     def search(self, state: State, depth: int, turn: int, maximizing_player):
#         """Start the MiniMax algorithm.
#         :param state: The state to start from.
#         :param depth: The maximum allowed depth for the algorithm.
#         :param turn: The turn count for the algorithm
#         :param maximizing_player: Whether this is a max node (True) or a min node (False).
#         :return: A tuple: (The min max algorithm value, The move in case of max node or None in min mode)
#         """
#
#         # if we reached the desired depth, then we evaluate the leaves: the evaluation of the leaves always return a
#         # 'None' move
#         if depth == 0:
#             return state.evaluation(turn), None
#         if state.less_than_3(turn):
#             return state.evaluation(turn), None
#
#         # if we still need to go more in depth in the search tree
#         optimal_minimax = None
#         optimal_move = None
#
#         # debug_data_matching(state.board, state.my_pos, state.rival_pos)
#
#         # compute successor_moves_list, which is list of tuples: (soldier_moving, neighbor, rival_cell_remove).
#         # Each item of the list represent a possible successor_move: each one will be then evaluated and the leaves expanded.
#         if turn < 18:
#             successor_moves_list = state.successor_moves_list_1st_stage(maximizing_player)
#         elif turn >= 18:
#             successor_moves_list = state.successor_moves_list_2nd_stage(maximizing_player)
#
#         # debug_data_matching(state.board, state.my_pos, state.rival_pos)
#
#         # here is implemented the true DFS dynamics
#         for successor_move in successor_moves_list:
#             # copy the state of the level above and update the copy
#             succ_state = State(np.copy(state.board), np.copy(state.my_pos), np.copy(state.rival_pos), state.regular_heuristic)
#             succ_state.get_move(successor_move, turn, maximizing_player)
#             # debug_data_matching(succ_state.board, succ_state.my_pos, succ_state.rival_pos)
#             search_tree = Minimax()
#             # debug_data_matching(state.board, state.my_pos, state.rival_pos)
#             # evaluate via recursive search function
#             # the minimizing_player is inverted because we will now expand/evaluate the moves by tho other player
#             minimax_value = search_tree.search(succ_state, depth=depth - 1, turn=turn + 1,
#                                                maximizing_player=not maximizing_player)[0]
#
#             # In this section we look for the optimal value (minimax) among all the children state (derived from successor_moves).
#             # We set the first value for optimal_minimax, and for optimal_move if it is my player's move
#             if optimal_minimax is None:
#                 optimal_minimax = minimax_value
#                 if maximizing_player:
#                     optimal_move = successor_move
#             # compare the minimax_value of the currently evaluated move and set it as optimal_minimax in case it is better
#             elif optimal_minimax is not None and minimax_value is not None:
#                 # print(minimax_value, optimal_minimax)
#                 if maximizing_player and minimax_value > optimal_minimax:
#                     optimal_minimax = minimax_value
#                     optimal_move = successor_move
#                 elif (not maximizing_player) and minimax_value < optimal_minimax:
#                     optimal_minimax = minimax_value
#
#         # return optimal minimax value and optimal move
#         return optimal_minimax, optimal_move


class AlphaBeta(SearchAlgos):
    def __init__(self):
        SearchAlgos.__init__(self)
        pass

    def search(self, state, depth, turn, maximizing_player, Alpha, Beta):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param beta: beta value
        :return A tuple: (The min max algorithm value, The move in case of max node or None in min mode)
        """
        # TODO: erase the following line and implement this function.
        if depth == 0:
            return state.evaluation(turn), None
        if state.less_than_3(turn):
            return state.evaluation(turn), None
        if state.CantMove(turn,maximizing_player):
            return state.evaluation(turn), None
        state.is_mill_closed=0
        # if we still need to go more in depth in the search tree
        optimal_minimax = None
        optimal_move = None
        CurMove = optimal_move
        # debug_data_matching(state.board, state.my_pos, state.rival_pos)
        if maximizing_player:
            CurMinMax = -np.inf
            if turn < 18:
                successor_moves_list = state.successor_moves_list_1st_stage(maximizing_player)
            elif turn >= 18:
                successor_moves_list = state.successor_moves_list_2nd_stage(maximizing_player)
            for successor_move in successor_moves_list:
                # copy the state of the level above and update the copy
                succ_state = State(np.copy(state.board), np.copy(state.my_pos), np.copy(state.rival_pos),
                                   state.regular_heuristic)
                succ_state.get_move(successor_move, turn, maximizing_player)
                # debug_data_matching(succ_state.board, succ_state.my_pos, succ_state.rival_pos)
                search_tree = AlphaBeta()
                # debug_data_matching(state.board, state.my_pos, state.rival_pos)
                minimax_value = search_tree.search(succ_state, depth - 1, turn + 1,
                                                   not maximizing_player, Alpha, Beta)[0]
                if minimax_value > CurMinMax:
                    CurMinMax, CurMove = minimax_value, successor_move
                Alpha = max(CurMinMax, Alpha)
                if CurMinMax >= Beta:
                    return (np.inf, CurMove)  # Which move to return here??
            return CurMinMax, CurMove
        ##
        if not maximizing_player:
            CurMinMax = np.inf
            if turn < 18:
                successor_moves_list = state.successor_moves_list_1st_stage(maximizing_player)
            elif turn >= 18:
                successor_moves_list = state.successor_moves_list_2nd_stage(maximizing_player)
            for successor_move in successor_moves_list:
                # copy the state of the level above and update the copy
                succ_state = State(np.copy(state.board), np.copy(state.my_pos), np.copy(state.rival_pos),
                                   state.regular_heuristic)
                succ_state.get_move(successor_move, turn, maximizing_player)
                # debug_data_matching(succ_state.board, succ_state.my_pos, succ_state.rival_pos)
                search_tree = AlphaBeta()
                # debug_data_matching(state.board, state.my_pos, state.rival_pos)
                minimax_value = search_tree.search(succ_state, depth - 1, turn + 1,
                                                   not maximizing_player, Alpha, Beta)[0]
                if minimax_value < CurMinMax:
                    CurMinMax, CurMove = minimax_value, successor_move
                Beta = min(CurMinMax, Beta)
                if CurMinMax <= Alpha:
                    return (-np.inf, CurMove)  # Which move to return here?
            return CurMinMax, CurMove
"""

---------------------------------------------------------------------- 
                        NEW MINIMAX
----------------------------------------------------------------------
"""
class Minimax(SearchAlgos):
    def __init__(self):
        SearchAlgos.__init__(self)
        pass

    def search(self, state, depth, turn, maximizing_player):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param beta: beta value
        :return A tuple: (The min max algorithm value, The move in case of max node or None in min mode)
        """
        # TODO: erase the following line and implement this function.
        if depth == 0:
            return state.evaluation(turn), None
        if state.less_than_3(turn):
            return state.evaluation(turn), None
        state.is_mill_closed=0
        # if we still need to go more in depth in the search tree
        optimal_minimax = None
        optimal_move = None
        CurMove = optimal_move
        # debug_data_matching(state.board, state.my_pos, state.rival_pos)
        # if turn==12:
        #     print("stop here")
        if maximizing_player:
            CurMinMax = -np.inf
            if turn < 18:
                successor_moves_list = state.successor_moves_list_1st_stage(maximizing_player)
            elif turn >= 18:
                successor_moves_list = state.successor_moves_list_2nd_stage(maximizing_player)
            for successor_move in successor_moves_list:
                # copy the state of the level above and update the copy
                succ_state = State(np.copy(state.board), np.copy(state.my_pos), np.copy(state.rival_pos),
                                   state.regular_heuristic)
                succ_state.get_move(successor_move, turn, maximizing_player)
                # debug_data_matching(succ_state.board, succ_state.my_pos, succ_state.rival_pos)
                search_tree = Minimax()
                # debug_data_matching(state.board, state.my_pos, state.rival_pos)
                minimax_value = search_tree.search(succ_state, depth - 1, turn + 1,
                                                   not maximizing_player)[0]
                if minimax_value > CurMinMax:
                    CurMinMax, CurMove = minimax_value, successor_move

            return CurMinMax, CurMove
        ##
        if not maximizing_player:
            CurMinMax = np.inf
            if turn < 18:
                successor_moves_list = state.successor_moves_list_1st_stage(maximizing_player)
            elif turn >= 18:
                successor_moves_list = state.successor_moves_list_2nd_stage(maximizing_player)
            for successor_move in successor_moves_list:
                # copy the state of the level above and update the copy
                succ_state = State(np.copy(state.board), np.copy(state.my_pos), np.copy(state.rival_pos),
                                   state.regular_heuristic)
                succ_state.get_move(successor_move, turn, maximizing_player)
                # debug_data_matching(succ_state.board, succ_state.my_pos, succ_state.rival_pos)
                search_tree = Minimax()
                # debug_data_matching(state.board, state.my_pos, state.rival_pos)
                minimax_value = search_tree.search(succ_state, depth - 1, turn + 1,
                                                   not maximizing_player)[0]
                if minimax_value < CurMinMax:
                    CurMinMax, CurMove = minimax_value, successor_move

            return CurMinMax, CurMove




"""
---------------------------------------------------------------------- 
STATE CLASS
----------------------------------------------------------------------
"""
class State:
    def __init__(self, board=None, my_pos=None, rival_pos=None, regular_heuristic=True):
        """
        :param board: It is a numpy.array of dimension 24
        :param my_pos: soldiers' position of the moving player
        :param rival_pos: soldiers' position of the moving player' rival
        player_number= player number to move 1 OR 2
        maximizing_player=True if its we False if it's rival (probably not needed)
        """
        self.board = board
        self.my_pos = my_pos
        self.rival_pos = rival_pos
        self.is_mill_closed = 0
        self.regular_heuristic = regular_heuristic

    # def __hash__(self):
    #     return hash((self.board, self.my_pos, self.rival_pos))

    def get_move(self, move: tuple, turn: int, maximizing_player: bool):
        """
        It computes the state from the successor tuple
        :param turn: turn of the game completed at this stage of the search
        :param move: (soldier_moving, neighbor, rival_cell_remove)
        :param maximizing_player: True if the move is by the rival player, False if the move is by my player
        :return: None
        """
        # Explanation of successor components
        # successor[0] = pos = neighboring cell in board
        # successor[1] = soldier = soldier moving in the turn
        # successor[2] = rival_dead_pos = rival cell soldier to be removed

        # debug_data_matching(self.board, self.my_pos, self.rival_pos)
        # if not debug_data_matching:
        #     print("stop here we got a problem")

        # if my player moves (player_id = 1)
        if maximizing_player:
            # unpack the move
            pos, soldier, rival_dead_pos = move
            # phase 1 of the game
            if turn < 18:
                self.board[pos] = 1
                self.my_pos[soldier] = pos
            # phase 2 of the game
            else:
                prev_pos = self.my_pos[soldier]
                self.board[prev_pos] = 0
                self.board[pos] = 1
                self.my_pos[soldier] = pos
            # if my player achieved a mill:
            if rival_dead_pos != -1:
                self.board[rival_dead_pos] = 0
                try:
                    dead_soldier = int(np.where(rival_dead_pos == self.rival_pos)[0][0])
                    self.is_mill_closed = 1
                except:
                    print("we got a problem here")
                    # print(f"board={self.board}")
                    # print(f"rival position={self.rival_pos}")
                    # print(f"my_pos ={self.my_pos}")
                    # print(f"rival_dead_pos={rival_dead_pos}")
                self.rival_pos[dead_soldier] = -2

        # else if rival player moves (player_id = 2)
        elif not maximizing_player:
            # unpack the move
            rival_pos, rival_soldier, my_dead_pos = move
            # phase 1 of the game
            if turn < 18:
                self.board[rival_pos] = 2
                self.rival_pos[rival_soldier] = rival_pos
            # phase 2 of the game
            else:
                rival_prev_pos = self.rival_pos[rival_soldier]
                self.board[rival_prev_pos] = 0
                self.board[rival_pos] = 2
                self.rival_pos[rival_soldier] = rival_pos
            # if the rival ha achieved a mill:
            if my_dead_pos != -1:
                self.board[my_dead_pos] = 0
                dead_soldier = int(np.where(self.my_pos == my_dead_pos)[0][0])
                self.my_pos[dead_soldier] = -2
                self.is_mill_closed = -1

    def successor_moves_list_1st_stage(self, maximizing_player: bool):
        """
        Computes all possible next moves for the player moving.
        Value of maximizing_player:
        True if the successors depend on a move by our player (id = 1),
        False if the successor_moves depend on a move by the rival player (id = 2).
        :param maximizing_player: True for our player's successor_moves, False for the rival's successor_moves
        :return: list of successor moves, where successor = tuple(soldier_moving, cell, rival_cell_remove)
        """
        successor_moves_list = list()
        agent_soldiers = np.full(9, -1)
        agent = None
        rival = None
        if maximizing_player is True:
            agent = 1
            rival = 2
            agent_soldiers = self.my_pos
        elif maximizing_player is False:
            agent = 2
            rival = 1
            agent_soldiers = self.rival_pos

        # part b - creating successor_moves_list list
        # choose the first soldier that is not in the game
        try:
            soldier_moving = np.where(agent_soldiers == -1)[0][0]
        except:
            print("stop here we got a problem")
        # find all empty positions in the board
        for cell in (np.where(self.board == 0))[0]:
            # if agent does not make a mill
            if not self.is_mill(cell, agent, self.board):
                successor_moves_list.append((cell, soldier_moving, -1))

            # else if agent does make a mill
            else:
                for rival_cell_remove in list(np.where(self.board == rival)[0]):
                    successor_moves_list.append((cell, soldier_moving, rival_cell_remove))

        return successor_moves_list

    def successor_moves_list_2nd_stage(self, maximizing_player: bool):
        """
        Computes all possible next moves for the player moving.
        Value of maximizing_player:
        True if the successors depend on a move by our player (id = 1),
        False if the successor_moves depend on a move by the rival player (id = 2).
        :param maximizing_player: True for our player's successor_moves, False for the rival's successor_moves
        :return: list of successor moves, where successor = tuple(soldier_moving, cell, rival_cell_remove)
        """
        # part a - define the variables:
        successor_moves_list = list()
        agent_soldiers = np.full(9, -1)
        agent = None
        rival = None
        if maximizing_player is True:
            agent = 1
            rival = 2
            agent_soldiers = self.my_pos
        elif maximizing_player is False:
            agent = 2
            rival = 1
            agent_soldiers = self.rival_pos

        # part b - creating successor_moves_list list
        # find indices of soldiers still in the game
        for soldier_moving in list(np.where(agent_soldiers >= 0)[0]):
            # finding an empty neighbors cells
            soldier_moving_cell = agent_soldiers[soldier_moving]
            # neighbors is the list of all possible child states, no 'collision check' done yet
            neighbors_list = get_directions(soldier_moving_cell)
            for neighbor in neighbors_list:
                # if the neighboring cell is free
                if self.board[neighbor] == 0:
                    # if the move does not complete a mill
                    temp_board = np.copy(self.board)
                    temp_board[soldier_moving_cell] = 0

                    if not self.is_mill(neighbor, agent, temp_board):
                        successor_moves_list.append((neighbor, soldier_moving, -1))
                    # else it does complete a mill
                    else:
                        # TODO: board here is broken;
                        for rival_cell_remove in list(np.where(temp_board == rival)[0]):
                            successor_moves_list.append((neighbor, soldier_moving, rival_cell_remove))
        return successor_moves_list

    def is_player(self, player, pos1, pos2, board):
        """
        Function to check if 2 positions have the player on them
        :param player: 1/2
        :param pos1: position
        :param pos2: position
        :return: boolean value
        """
        if board[pos1] == player and board[pos2] == player:
            return True
        else:
            return False

    def check_next_mill(self, position, player, board):
        """
        Function to check if a player can make a mill in the next move.
        :param position: curren position
        :param player: 1/2
        :return:
        """
        mill = [
            (self.is_player(player, 1, 2, board) or self.is_player(player, 3, 5, board)),
            (self.is_player(player, 0, 2, board) or self.is_player(player, 9, 17, board)),
            (self.is_player(player, 0, 1, board) or self.is_player(player, 4, 7, board)),
            (self.is_player(player, 0, 5, board) or self.is_player(player, 11, 19, board)),
            (self.is_player(player, 2, 7, board) or self.is_player(player, 12, 20, board)),
            (self.is_player(player, 0, 3, board) or self.is_player(player, 6, 7, board)),
            (self.is_player(player, 5, 7, board) or self.is_player(player, 14, 22, board)),
            (self.is_player(player, 2, 4, board) or self.is_player(player, 5, 6, board)),
            (self.is_player(player, 9, 10, board) or self.is_player(player, 11, 13, board)),
            (self.is_player(player, 8, 10, board) or self.is_player(player, 1, 17, board)),
            (self.is_player(player, 8, 9, board) or self.is_player(player, 12, 15, board)),
            (self.is_player(player, 3, 19, board) or self.is_player(player, 8, 13, board)),
            (self.is_player(player, 20, 4, board) or self.is_player(player, 10, 15, board)),
            (self.is_player(player, 8, 11, board) or self.is_player(player, 14, 15, board)),
            (self.is_player(player, 13, 15, board) or self.is_player(player, 6, 22, board)),
            (self.is_player(player, 13, 14, board) or self.is_player(player, 10, 12, board)),
            (self.is_player(player, 17, 18, board) or self.is_player(player, 19, 21, board)),
            (self.is_player(player, 1, 9, board) or self.is_player(player, 16, 18, board)),
            (self.is_player(player, 16, 17, board) or self.is_player(player, 20, 23, board)),
            (self.is_player(player, 16, 21, board) or self.is_player(player, 3, 11, board)),
            (self.is_player(player, 12, 4, board) or self.is_player(player, 18, 23, board)),
            (self.is_player(player, 16, 19, board) or self.is_player(player, 22, 23, board)),
            (self.is_player(player, 6, 14, board) or self.is_player(player, 21, 23, board)),
            (self.is_player(player, 18, 20, board) or self.is_player(player, 21, 22, board))
        ]

        return mill[position]

    def is_mill(self, position, player_index, board):
        """
        Return True if a player has a mill on the given position
        :param player_index: player index
        :param position: 0-23
        :return: bool
        """
        return self.check_next_mill(position, player_index, board)
    def CantMove(self,turn,maximizing_player):
        Num_of_blocked_soldiers=0
        if turn>18:
            if maximizing_player:
                Num_soldiers_in_play = len(np.where(self.my_pos >= 0)[0])
                for soldier in self.my_pos:
                    if soldier >= 0 and not self.check_non_blocked_soldier(soldier):
                        Num_of_blocked_soldiers += 1
                if Num_of_blocked_soldiers==Num_soldiers_in_play:
                    return True
            else:
                Num_soldiers_in_play = len(np.where(self.rival_pos >= 0)[0])
                for soldier in self.rival_pos:
                    # if soldier is sill to be placed
                    if soldier >= 0 and not self.check_non_blocked_soldier(soldier):
                        Num_of_blocked_soldiers += 1
                if Num_of_blocked_soldiers == Num_soldiers_in_play:
                    return True
        return False



    def evaluation(self, turn: int):
        # TODO: these weights for the evaluation functions are optimized for a game with the third phase, we don't play 
        #  the third phase here, so we should recompute the optiml values
        if self.regular_heuristic:
            coefficient_array = np.full(8, 0)
            if turn < 18:
                coefficient_array = np.array([18, 25, 1, 9, 10, 7, 0, 0])
            elif turn >= 18:
                coefficient_array = np.array([80, 25, 20, 11, 0, 0, 8, 1086]) # C1=80
            if turn >= 40:
                coefficient_array = np.array([80, 15, 10, 3, 0, 0, 8, 1086])  # C1=80
            # if turn >= 40:
            #     coefficient_array = np.array([30, 150, 20, 11, 0, 0, 8, 0])
            # 60, 20, 0, 10, 0, 0, 0, 0
            # if turn >= 30:
                # coefficient_array = np.array([60, 20, 0, 10, 0, 0, 0, 0])

            # elif turn >= 200:
            #     coefficient_array = np.array([270,0,0,0,0,0,0,0])
                # coefficient_array = np.array([14, 43, 10, 11, 5, 0.5, 8, 1086])

            # if turn >= 25:
            #     coefficient_array = np.array([25, 35, 7, 11, 5, 0.5, 8, 1150])
            # compute the values of the various heuristics
            r1 = self.r1()
            r2, r5, r6, r7 = self.r2r5r6r7()
            r3, r8 = self.r3r8()
            r4 = self.r4()
            # r8=0
            # r8=0 # something strange with it...
            # r8 = self.r8()
            r_array = np.array([r1, r2, r3, r4, r5, r6, r7, r8])
            return (np.multiply(coefficient_array, r_array)).sum()
        else:
            # coefficient_array = np.full(8, 0)
            # if turn < 18:
            # coefficient_array = np.array([18, 26, 1, 9, 10, 7, 0, 0])
            # elif turn >= 18:
            # coefficient_array = np.array([30, 31, 10, 11, 5, 0.5, 8, 1086])
            coefficient_array = np.array([60, 20, 10])  # Like average of two stage coefficients
            # if turn >= 25:
            #     coefficient_array = np.array([25, 35, 7, 11, 5, 0.5, 8, 1150])
            # compute the values of the various heuristics
            r1 = self.r1()
            # r1=0
            # r2, r5, r6, r7 = self.r2r5r6r7()
            # r3, r8 = self.r3r8()
            r2 = self.r2_light()
            r2 = 0
            r4 = self.r4()
            # r8 = self.r8()
            r_array = np.array([r1, r2, r4])
            return (np.multiply(coefficient_array, r_array)).sum()

    # TODO: implement R1 (see notes.txt for advice)
    def r1(self):
        """
        Returns:
        1 if the player closed a mill in the last move;
        -1 if the rival closed a mill in the last move;
        0 otherwise.
        :return: int, values are (1, 0, -1)
        """
        # if self.is_mill_closed==1:
        #     print("----- self mill is 1 -------")
        # if self.is_mill_closed == 0:
        #     print("----- self mill is 0 -------")
        return self.is_mill_closed

    # TODO: it counts twice when there is a cross-shaped or a T-shaped of two 2-piece configurations, while there is
    #  actually only one piece missing
    def r2r5r6r7(self):
        """
        Computes evaluation functions R2, R6, R7.
        R2 counts the number of closed mills.
        R5 counts the number of 2-piece configurations (a piece can be added in a cell to close a mill).
        R6 counts the number of 3-piece configurations (a piece can be added in two different positions to close a morris).
        R7 counts the number of double mills.
        :return: r2, r6, r7 (tuple of three int items) 
        """
        mill_combinations = [[0, 1, 2], [0, 3, 5], [2, 4, 7], [5, 6, 7], [1, 9, 17], [8, 9, 10], [10, 12, 15],
                             [13, 14, 15], [8, 11, 13], [16, 17, 18], [18, 20, 23], [16, 19, 21], [21, 22, 23],
                             [3, 11, 19], [20, 12, 4], [6, 14, 22]]
        mine_mill = list()
        rival_mill = list()
        mine_tris = list()
        rival_tris = list()
        r2 = 0
        r5 = 0
        r6 = 0
        r7 = 0

        # R2, R6
        for mill in mill_combinations:
            # R2
            if self.board[mill[0]] == self.board[mill[1]] == self.board[mill[2]]:
                if self.board[mill[0]] == 1:
                    r2 += 1
                    mine_mill = mine_mill + mill
                elif self.board[mill[0]] == 2:
                    r2 -= 1
                    rival_mill = rival_mill + mill
            # R6 first part
            # tris is a 2-piece configuration, i.e. when you have two soldiers and a void cell in a row.
            # We collect them in two lists the same way we do with mills.
            else:
                # tris contains the values of three cells in a row.
                # mine_tris and rival_tris contain a list of cells where soldiers of player or rival are and are
                # candidate tris for the 3-piece configurations
                tris = self.board[mill].tolist()
                # if it is a candidate for the player
                if tris.count(1) == 2 and tris.count(0) == 1:
                    index_0 = tris.index(0)
                    mill.pop(index_0)
                    mine_tris = mine_tris + mill
                # if it is a candidate for the rival
                elif tris.count(2) == 2 and tris.count(0) == 1:
                    index_0 = tris.index(0)
                    mill.pop(index_0)
                    rival_tris = rival_tris + mill
        # R5 first part
        # mine_tris is a list containing all the cells included in 2-piece configurations: to each configuration,
        # 2 items correspond. To compute a preliminary value of 2-piece configurations we will divide the length of the
        # lists by 2. Later in the method, we will subtract twice the number of 3-piece configurations to the number of
        # 2-piece configurations computed as preliminary value.
        r5 = len(mine_tris) / 2 - len(rival_tris) / 2

        # R7
        # mine_mill and rival_mill are two lists, each one long n*3 items, where n is the number of mills present. If
        # an element belongs to two mills, it should appear twice in the list.
        for cell in range(24):
            if mine_mill.count(cell) > 1:
                r7 += 1
            elif rival_mill.count(cell) > 1:
                r7 -= 1

        # R6 second part
        for cell in range(24):
            if mine_tris.count(cell) == 2:
                r6 += 1
            elif rival_tris.count(cell) == 2:
                r6 -= 1
            elif mine_tris.count(cell) > 2 or rival_tris.count(cell) > 2:
                raise ValueError

        # R5 second part
        # r5 = r5 - 2 * r6

        return r2, r5, r6, r7

    def r3r8(self):
        """
        Computes R3 and R8 evaluation functions.
        R3: number of blocked soldiers of the rival minus number of blocked soldiers of my player.
        R8: returns an int depending on the state: if the state is winning for my player (1), or for rival (-1),
        or otherwise (0).
        :return: R3, R8
        """
        # initialize variables for the enumeration
        r3 = 0
        r8 = 0
        my_n_blocked_soldiers = 0
        my_n_non_placed_soldiers = 0
        rival_n_blocked_soldiers = 0
        rival_n_non_placed_soldiers = 0

        # R3
        # Computation for player -------------
        # check for each soldier in the game if it can move
        for soldier in self.my_pos:
            # if soldier is sill to be placed
            if soldier == -1:
                my_n_non_placed_soldiers += 1
            # if soldier is on board and is blocked
            elif soldier >= 0 and not self.check_non_blocked_soldier(soldier):
                my_n_blocked_soldiers += 1
        # Computation for rival --------------
        # check for each soldier in the game if it can move
        for soldier in self.rival_pos:
            # if soldier is still to be placed
            if soldier == -1:
                rival_n_non_placed_soldiers += 1
            # if soldier is on board and is blocked
            elif soldier >= 0 and not self.check_non_blocked_soldier(soldier):
                rival_n_blocked_soldiers += 1
        # compute R3 value
        r3 = rival_n_blocked_soldiers - my_n_blocked_soldiers

        # R8
        # collects np.arrays of bool: True if the soldier is alive, False if it is dead.
        # Then it counts how many non_zero elements (non-False, i.e. alive soldiers) there are.
        my_soldiers_alive = np.count_nonzero(self.my_pos >= 0)
        rival_soldiers_alive = np.count_nonzero(self.rival_pos >= 0)

        if my_soldiers_alive < 3:
            r8 = -1
        elif rival_soldiers_alive < 3:
            r8 = 1

        # if my player have 3 or more soldiers on the field or to be placed but they are blocked
        elif my_soldiers_alive - my_n_blocked_soldiers == 0:  # WAS:  my_soldiers_alive - my_n_blocked_soldiers - my_n_non_placed_soldiers <= 0:
            r8 = -1
        elif rival_soldiers_alive - rival_n_blocked_soldiers == 0:  # WAS :  rival_soldiers_alive - my_n_blocked_soldiers - my_n_non_placed_soldiers <= 0:
            r8 = 1

        return r3, r8

    def r4(self):
        """
        Compute material difference
        :return: n. of our soldiers alive - n. of rival's soldiers alive
        """
        return len(np.where(self.my_pos >= 0)[0]) - len(np.where(self.rival_pos >= 0)[0])

    def r2_light(self):
        morris_combinations = [[0, 1, 2], [0, 3, 5], [2, 4, 7], [5, 6, 7], [1, 9, 17], [8, 9, 10], [10, 12, 15],
                               [13, 14, 15], [8, 11, 13], [16, 17, 18], [18, 20, 23], [16, 19, 21], [21, 22, 23],
                               [3, 11, 19], [20, 12, 4], [6, 14, 22]]
        r2 = 0
        for morris in morris_combinations:
            if self.board[morris[0]] == self.board[morris[1]] == self.board[morris[2]]:
                if self.board[morris[0]] == 1:
                    r2 += 1
                else:
                    r2 -= 1
        return r2

    def check_non_blocked_soldier(self, cell: int) -> bool:
        """
        Checks if the soldier in cell can move.
        :param cell: int - the cell where the soldier is
        :return: True if it can move, False if it is stuck
        """
        for neighboring_cell in get_directions(cell):
            if self.board[neighboring_cell] == 0:
                return True
        return False

    def less_than_3(self, turn):
        if turn >= 18:
            if np.sum(self.my_pos >= 0) < 3 or np.sum(self.rival_pos >= 0) < 3:
                return True
        return False
