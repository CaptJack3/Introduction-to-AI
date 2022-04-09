import operator
import numpy as np
import os

#TODO: edit the alpha and beta initialization values for AlphaBeta algorithm.
# instead of 'None', write the real initialization value, learned in class.
# hint: you can use np.inf
ALPHA_VALUE_INIT = None
BETA_VALUE_INIT = None 

def get_directions(position):
    """Returns all the possible directions of a player in the game as a list.
    """
    assert 0 <= position <= 23, "illegal move"
    adjacent = [
        [1, 3],
        [0, 2, 9],
        [1, 4],
        [0, 5, 11],
        [2, 7, 12],
        [3, 6],
        [5, 7, 14],
        [4, 6],
        [9, 11],
        [1, 8, 10, 17],
        [9, 12],
        [3, 8, 13, 19],
        [4, 10, 15, 20],
        [11, 14],
        [6, 13, 15, 22],
        [12, 14],
        [17, 19],
        [9, 16, 18],
        [17, 20],
        [11, 16, 21],
        [12, 18, 23],
        [19, 22],
        [21, 23, 14],
        [20, 22]
    ]
    return adjacent[position]

# def tup_add(t1, t2):
#     """
#     returns the sum of two tuples as tuple.
#     """
#     return tuple(map(operator.add, t1, t2))

def printBoard(board):
    print(int(board[0]),"(00)-----------------------",int(board[1]),"(01)-----------------------",int(board[2]),"(02)")
    print("|                             |                             |")
    print("|                             |                             |")
    print("|                             |                             |")
    print("|       ",int(board[8]),"(08)--------------",int(board[9]),"(09)--------------",int(board[10]),"(10)   |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       |        ",int(board[16]),"(16)-----",int(board[17]),"(17)-----",int(board[18]),"(18)   |        |")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print(int(board[3]),"(03)-",int(board[11]),"(11)---",int(board[19]),"(19)                 ",
          int(board[20]),"(20)-",int(board[12]),"(12)---",int(board[4]),"(04)")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print("|       |        ",int(board[21]),"(21)-----",int(board[22]),"(22)-----",int(board[23]),"(23)   |        |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       ",int(board[13]),"(13)--------------",int(board[14]),"(14)--------------",int(board[15]),"(15)   |")
    print("|                             |                             |")
    print("|                             |                             |")
    print("|                             |                             |")
    print(int(board[5]),"(05)-----------------------",int(board[6]),"(06)-----------------------",int(board[7]),"(07)")
    print("\n")

