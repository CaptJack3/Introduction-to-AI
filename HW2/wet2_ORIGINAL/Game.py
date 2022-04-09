import numpy as np
import os
import random
import time
import utils

class Game:
    def __init__(self, board, players_positions):
        """Initialize the game properties with parameters.
        input:
            - board: np.array. The initial board.
              board size is 24 (0-23). 0- free space. 1= player1, 2- player2
            - players_positions: the initial players positions
              list of 2 np.arrays of size 9.
              players_positions[0]- player1 positions, players_positions[1]- player2 positions,
              0-23- place on board, -1- unplaced piece, -2- dead piece.
        """
        #assert len(players_positions) == 18, 'Supporting 18 players only' #!!!
        self.map = board

        self.players_positions = players_positions
        self.players_score = [0 ,0] # init scores for each player
        self.directions = utils.get_directions #!!!

        self.turn = 0

    def update_staff_with_pos(self, move:tuple):
        """
        :param move: (pos, soldier, dead_opponent_pos).
        :param pos: np.array of player's new positions.
        :param soldier: soldier that moved 0-8
        """
        next_pos, soldier, dead_opponent = move

        prev_pos = self.players_positions[self.turn][soldier]
        player_id = self.map[prev_pos]
        self.map[prev_pos] = 0
        self.map[next_pos] = player_id
        self.players_positions[self.turn][soldier] = next_pos

        if dead_opponent != -1:
            self.remove_pos_from_board_and_update_position(dead_opponent)

        self.turn = 1 - self.turn

    def add_pos_to_board_and_update_position(self, move, player_idx):
        """
        :param pos: 0-23
        :param player_idx: 0/1
        :return:
        """
        pos, soldier, dead_opponent = move

        self.map[pos] = player_idx + 1
        self.players_positions[player_idx][soldier] = pos

        if dead_opponent != -1:
            self.remove_pos_from_board_and_update_position(dead_opponent)

        self.turn = 1 - self.turn

    def remove_pos_from_board_and_update_position(self, opponent_pos):
        player_idx = int(self.map[opponent_pos]) - 1
        self.map[opponent_pos] = 0
        soldier = np.where(self.players_positions[player_idx] == opponent_pos)[0][0]
        self.players_positions[player_idx][soldier] = -2

    def player_cant_move(self, player_id)->bool:
        """
        :param player_id: 0/1
        :return:
        """
        player_pos = self.get_player_position(player_id)
        all_next_positions = [(player_current_pos, player_next_pos)
                              for player_current_pos in player_pos[0]
                              for player_next_pos in self.directions(player_current_pos)]
        possible_next_positions = [pos for pos in all_next_positions if self.pos_feasible_on_board(pos[1])]
        return len(possible_next_positions) == 0

    def pos_feasible_on_board(self, pos):
        """
        :param pos: 0-23
        :return: boolean value
        """
        # on board
        on_board = (0 <= pos < 24)
        if not on_board:
            return False
        
        # free cell
        value_in_pos = self.map[pos]
        free_cell = (value_in_pos == 0)
        return free_cell

    def check_move(self, move)->bool:
        """
        check_move before update board
        :param move: (pos, soldier, dead_opponent_pos)
        :return: boolean
        """
        if not self.pos_feasible_on_board(move[0]): # Check that the position is to an empty cell on board
            return False
        if self.players_positions[self.turn][move[1]] == -1: # If soldier not on board yet -> you can put him on the free cell
            return True
        if len(np.where(self.players_positions[self.turn] == -1)[0]) > 0: # if soldier is on board -> check that all soldiers are already on board.
            return False
        if self.players_positions[self.turn][move[1]] == -2:
            return False
        if not any(m == move[0] for m in self.directions(self.players_positions[self.turn][move[1]])):
            return False
        return True

    def print_board_to_terminal(self, player_id):
        board_to_print = self.get_map_for_player_i(player_id)
        utils.printBoard(board_to_print)

    def get_map_for_player_i(self, player_id):
        """
        :param player_id: 0 or 1
        :return: map_copy
        """
        map_copy = self.map.copy()

        pos_player_id = self.get_player_position(player_id)
        pos_second = self.get_player_position(1 - player_id)

        # flip positions
        for i in pos_player_id:
           map_copy[i] = 1
           for i in pos_second:
               map_copy[i] = 2
        return map_copy

    def get_player_position(self, player_id):
        """
        :param player_id: 0/1
        :return:
        """
        pos = np.where(self.map == player_id + 1)
        return pos

    def isPlayer(self, player, pos1, pos2):
        """
        Function to check if 2 positions have the player on them
        :param player: 1/2
        :param pos1: position
        :param pos2: position
        :return: boolean value
        """
        if (self.map[pos1] == player and self.map[pos2] == player):
            return True
        else:
            return False

    def checkNextMill(self, position, player):
        """
        Function to check if a player can make a mill in the next move.
        :param position: curren position
        :param board: np.array
        :param player: 1/2
        :return:
        """
        mill = [
            (self.isPlayer(player, 1, 2) or self.isPlayer(player, 3, 5)),
            (self.isPlayer(player, 0, 2) or self.isPlayer(player, 9, 17)),
            (self.isPlayer(player, 0, 1) or self.isPlayer(player, 4, 7)),
            (self.isPlayer(player, 0, 5) or self.isPlayer(player, 11, 19)),
            (self.isPlayer(player, 2, 7) or self.isPlayer(player, 12, 20)),
            (self.isPlayer(player, 0, 3) or self.isPlayer(player, 6, 7)),
            (self.isPlayer(player, 5, 7) or self.isPlayer(player, 14, 22)),
            (self.isPlayer(player, 2, 4) or self.isPlayer(player, 5, 6)),
            (self.isPlayer(player, 9, 10) or self.isPlayer(player, 11, 13)),
            (self.isPlayer(player, 8, 10) or self.isPlayer(player, 1, 17)),
            (self.isPlayer(player, 8, 9) or self.isPlayer(player, 12, 15)),
            (self.isPlayer(player, 3, 19) or self.isPlayer(player, 8, 13)),
            (self.isPlayer(player, 20, 4) or self.isPlayer(player, 10, 15)),
            (self.isPlayer(player, 8, 11) or self.isPlayer(player, 14, 15)),
            (self.isPlayer(player, 13, 15) or self.isPlayer(player, 6, 22)),
            (self.isPlayer(player, 13, 14) or self.isPlayer(player, 10, 12)),
            (self.isPlayer(player, 17, 18) or self.isPlayer(player, 19, 21)),
            (self.isPlayer(player, 1, 9) or self.isPlayer(player, 16, 18)),
            (self.isPlayer(player, 16, 17) or self.isPlayer(player, 20, 23)),
            (self.isPlayer(player, 16, 21) or self.isPlayer(player, 3, 11)),
            (self.isPlayer(player, 12, 4) or self.isPlayer(player, 18, 23)),
            (self.isPlayer(player, 16, 19) or self.isPlayer(player, 22, 23)),
            (self.isPlayer(player, 6, 14) or self.isPlayer(player, 21, 23)),
            (self.isPlayer(player, 18, 20) or self.isPlayer(player, 21, 22))
        ]

        return mill[position]

    def isMill(self, position, player_index):
        """
        Return True if a player has a mill on the given position
        :param position: 0-23
        :return:
        """
        return self.checkNextMill(position, player_index)

    def check_end_game(self, player_idx:int)->bool:
        dead = np.where(self.players_positions[player_idx] != -2)[0]
        if len(dead) < 3:
            return True
        return False
