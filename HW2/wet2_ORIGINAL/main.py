import argparse
from GameWrapper import GameWrapper
import os, sys
import utils
import numpy as np

if __name__ == "__main__":
    players_options = [x+'Player' for x in ['Live', 'Simple', 'Minimax', 'Alphabeta', 'GlobalTimeAB', 'LightAB',
                                            'HeavyAB', 'Compete']]

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-player1', default='RandomPlayer', type=str,
                        help='The type of the first player.',
                        choices=players_options)
    parser.add_argument('-player2', default='SimplePlayer',  type=str,
                        help='The type of the second player.',
                        choices=players_options)
    parser.add_argument('-move_time', default=250, type=float, 
                        help='Time (sec) for each turn.')
    parser.add_argument('-game_time', default=2000, type=float, 
                        help='Global game time (sec) for each player.')
    parser.add_argument('-terminal_viz', action='store_true',
                        help='Show game in terminal only.')

    args = parser.parse_args()

    # check validity of game and turn times
    if args.game_time < args.move_time:
        raise Exception('Wrong time arguments.')

    # Players inherit from AbstractPlayer - this allows maximum flexibility and modularity
    player_1_type = 'players.' + args.player1
    player_2_type = 'players.' + args.player2
    game_time = args.game_time
    __import__(player_1_type)
    __import__(player_2_type)
    player_1 = sys.modules[player_1_type].Player(game_time)
    player_2 = sys.modules[player_2_type].Player(game_time)

    # print game info to terminal
    print('Starting Game!')
    print(args.player1, 'VS', args.player2)
    print('Players have', args.move_time, 'seconds to make a single move.')
    print('Each player has', game_time, 'seconds to play in a game (global game time, sum of all moves).')

    # create game with the given args
    game = GameWrapper(player_1=player_1, player_2=player_2,players_positions=[np.full(9, -1),np.full(9, -1)],
                    print_game_in_terminal=True,
                    time_to_make_a_move=args.move_time, 
                    game_time=game_time)
    # start playing!
    game.run_game()

