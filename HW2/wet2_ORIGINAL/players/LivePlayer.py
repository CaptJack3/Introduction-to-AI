from players.AbstractPlayer import AbstractPlayer
import numpy as np
import os, sys

class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None             # and add more fields to Player
        self.my_pos = None
        self.rival_pos = None
        self.turn = 0

    def set_game_params(self, board):
        self.board = board
        self.my_pos = np.full(9, -1)
        self.rival_pos = np.full(9, -1)
        self.turn = 0

    def update_player_on_board(self, next_pos, prev_pos, soldier):
        # update position and board:
        self.board[next_pos] = 1
        self.board[prev_pos] = 0
        self.my_pos[soldier] = next_pos

    def make_mill_get_rival_cell(self) -> int:
        while True:
            rival_cell = int(input('Choose rival cell to take off board: '))
            if 0 <= rival_cell < 24:
                if self.board[rival_cell] == 2:
                    break
                print("rival is not in cell", rival_cell, "choose again")
            else:
                print("cell", rival_cell, "is out of board bounce")

        rival_idx = np.where(self.rival_pos == rival_cell)[0][0]
        self.rival_pos[rival_idx] = -2
        self.board[rival_cell] = 0
        return rival_cell

    def stage_1_move(self) -> tuple:
        while True:
            cell = int(input('Enter cell number: '))
            if 0 <= cell <= 23 and self.board[cell] == 0:
                break
            print("cell number", cell, "is invalid. Try again")

        soldier_that_moved = int(np.where(self.my_pos == -1)[0][0])
        self.my_pos[soldier_that_moved] = cell
        self.board[cell] = 1

        rival_cell = -1 if not self.is_mill(cell) else self.make_mill_get_rival_cell()

        return cell, soldier_that_moved, rival_cell

    def stage_2_move(self)->tuple:
        prev_cell = -1
        t = True
        while t:
            prev_cell = int(input('Choose cell to move from:'))
            if prev_cell in self.my_pos:
                #print(self.directions(prev_cell)) # !!!
                for direction in self.directions(prev_cell):
                    #print("direction, self.board[direction]", direction, self.board[direction]) # !!!
                    if int(self.board[direction]) == 0:
                        t = False
                        break
            if t:
                print("cant move from cell", prev_cell)
        while True:
            cell = int(input('Choose cell to move to:'))
            if int(self.board[cell]) == 0:
                if cell in self.directions(prev_cell):
                    break
            print("can't move to cell", cell, "from cell", prev_cell)

        soldier_that_moved = int(np.where(self.my_pos == prev_cell)[0][0])
        self.update_player_on_board(cell, prev_cell, soldier_that_moved)

        rival_cell = -1 if not self.is_mill(cell) else self.make_mill_get_rival_cell()
        return cell, soldier_that_moved, rival_cell

    def make_move(self, time_limit):
        #print("Live player: turn:", self.turn)

        if self.turn < 18:
            move = self.stage_1_move()
            self.turn += 1
            return move

        else:
            move = self.stage_2_move()
            self.turn += 1
            return move

    def set_rival_move(self, move):
        rival_pos, rival_soldier, my_dead_pos = move

        if self.turn < 18:
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos
        else:
            rival_prev_pos = self.rival_pos[rival_soldier]
            self.board[rival_prev_pos] = 0
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos
        if my_dead_pos != -1:
            self.board[my_dead_pos] = 0
            dead_soldier = int(np.where(self.my_pos == my_dead_pos)[0][0])
            self.my_pos[dead_soldier] = -2
        self.turn += 1

    def print_player_board(self):
        print("board:")
        print(self.board)
        print(np.arange(24))

        print("my pos:")
        print(self.my_pos)
        print(np.arange(9))