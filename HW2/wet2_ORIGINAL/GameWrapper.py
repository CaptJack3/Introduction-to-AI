from Game import Game
import numpy as np
import time
import sys

class GameWrapper:
    def __init__(self, player_1, player_2, players_positions,
                print_game_in_terminal,
                time_to_make_a_move=2, game_time=100):
        """Initialize the game wrapper and the initial board state with parameters.
        input:
            - player_1, player_2: players objects (such as LivePlayer object).
            - players_positions: the initial players positions
              list of 2 np.arrays of size 9.
              players_positions[0]- player1 positions, players_positions[1]- player2 positions,
              0-23- place on board, -1- unplaced piece, -2- dead piece.
            - print_game_in_terminal: bool. Show only the winner and final scores if false.
            - time_to_make_a_move: time for a single turn.
            - game_time: total time for each player's game.
        """

        # check that each Player implements the following methods:
        self.players = [player_1, player_2]
        for player in self.players:
            assert hasattr(player, 'set_game_params')
            assert hasattr(player, 'make_move')
            assert hasattr(player, 'set_rival_move')

        self.print_game_in_terminal = print_game_in_terminal
        self.time_to_make_a_move = time_to_make_a_move
        self.game_time_left_for_players = [game_time, game_time]

        initial_board = self.set_initial_board()
        self.players_positions = players_positions
        self.game = Game(initial_board, players_positions)

        for i, player in enumerate(self.players):
            player.set_game_params(self.game.get_map_for_player_i(player_id=i))


    def check_cant_move_end_game(self, player_index):
        """
          :param player_index: 0\1
          :return: Boolean
          """
        if self.game.player_cant_move(player_index):
            messages = [f'    Player {player_index} Won!']
            self.pretty_print_end_game(messages)
        else:
            return False

    def play_turn(self, player_index):
        """
        :param player_index: 0 or 1
        :return: move = (pos, soldier, dead_opponent_pos)
        """

        start = time.time()
        move = self.players[player_index].make_move(self.time_to_make_a_move) # move returns (pos, soldier, dead_opponent_pos)
        end = time.time()
        time_diff = end - start

        # reduce time from global time
        self.game_time_left_for_players[player_index] -= time_diff

        if time_diff > self.time_to_make_a_move or self.game_time_left_for_players[player_index] <= 0:
            player_index_time_up = player_index + 1
            messages = [f'Time Up For Player {player_index_time_up}',
                        f'    Player {3 - player_index_time_up} Won!']
            self.pretty_print_end_game(messages)

        assert self.game.check_move(move), 'illegal move'

        self.players[1 - player_index].set_rival_move(move)

        return move

    def run_game(self):
        self.turn_number = 0

        if self.print_game_in_terminal:
            print('\nInitial board:')
            self.game.print_board_to_terminal(player_id=0)

        while True:
            player_index = self.turn_number % 2 # is 0 or 1
            print('player', player_index + 1)
            if self.turn_number >= 18:
                self.check_cant_move_end_game(player_index)
            move = self.play_turn(player_index)
            if self.turn_number >= 18:
                self.game.update_staff_with_pos(move)
            else:
                self.game.add_pos_to_board_and_update_position(move=move, player_idx=player_index)

            made_mill = self.game.isMill(position=move[0], player_index=player_index + 1)

            if made_mill:
                assert move[2] != -1, 'illegal move, did not choose opponent soldier to take out'
            else:
                assert move[2] == -1, 'illegal move, chose opponent soldier to take out but did not make mill'

            if self.print_game_in_terminal:
                print('\nBoard after player', player_index + 1, 'moved')
                print("player", player_index + 1, "moved soldier to position", move[0])
                if made_mill:
                    print("the player made a mill and killed rival soldier from cell", move[2])
                self.game.print_board_to_terminal(player_id=0)

            if self.game.check_end_game(1 - player_index):
                # If rival has 3 players or less -> End Game!
                print("End game")
                messages = [f'    Player {player_index + 1} Won!']
                self.pretty_print_end_game(messages)

            self.turn_number += 1

    ################## helper functions ##################
    @staticmethod
    def set_initial_board():
        return np.zeros(24)
    
    @staticmethod
    def pretty_print_end_game(messages):
        print('####################')
        print('####################')
        for message in messages:
            print(message)
        print('####################')
        print('####################')
        sys.exit(0)