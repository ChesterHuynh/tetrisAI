# Reference: https://github.com/nuno-faria/tetris-ai/blob/master/tetris.py
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import random
import pdb

style.use("ggplot")

class Tetris:
    grid_colors = {'white': (255, 255, 255), \
              'grey': (128, 128, 128), \
              'black': (0, 0, 0)
              }

    piece_colors = [
        (255, 255, 255), # White
        (255, 0,   0  ), # Red
        (54,  175, 144), # Green
        (0,   0,   255), # Blue
        (254, 151, 32 ), # Orange
        (255, 255, 0  ), # Yellow
        (147, 88,  254), # Purple
        (102, 217, 238) # Cyan
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    GRID_HEIGHT = 20
    GRID_WIDTH = 10

    score = level = lines_cleared = 0

    def __init__(self):
        """
        Initialize a Tetris game
        """
        self.reset_game()


    def reset_game(self):
        """
        Reset a game
        """
        self.board = [[0] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        self.score = self.level = self.lines = 0
        self.ind = random.randrange(len(self.pieces))
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {'x': self.GRID_WIDTH//2 - len(self.piece[0])//2,
                            'y': 0
                            }
        self.gameover = False
        return self.get_state_props(self.board)

    def rotate_CW(self, piece, pos):
        """
        Rotate the currently active piece clockwise
        """
        num_rows_orig = num_cols_new = len(piece)
        num_cols_orig = num_rows_new = len(piece[0])
        rotated_array = []

        if pos['x'] + len(piece) > self.GRID_WIDTH:
            pos['x'] = self.GRID_WIDTH - len(piece)
        if pos['y'] + len(piece[0]) > self.GRID_HEIGHT:
            pos['y'] = self.GRID_HEIGHT - len(piece[0])

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig-1)-j][i]
            rotated_array.append(new_row)
        piece = rotated_array
        return piece, pos

    def get_state_props(self, board):
        """
        Get properties of the current state of the board
        """
        lines_cleared = self.check_cleared_rows()
        holes = self.count_holes(board)
        total_bumpiness, max_bumpiness = self.bumpiness(board)
        sum_height, max_height, min_height = self.compute_height(board)
        return [lines_cleared, holes, total_bumpiness, sum_height]

    def count_holes(self, board):
        """
        We count the number of times there's a non-zero entry above a 0
        """
        board = np.array(board)
        currs = board[:-1] # All rows except the last one
        belows = board[1:] # All rows except the first one

        # Take the difference of each row and its corresponding row below it
        diffs = currs - belows

        # If there was a 0 below a given element in currs, then diff should
        # have the same value as currs at that element
        num_holes = np.sum(diffs == currs)
        return num_holes

    def bumpiness(self, board):
        """
        We total the differences in heights between each column of the board.
        The height of a column is the nonzero value at the highest point in the
        column.
        """
        board = np.array(board)
        mask = board != 0

        # Get the inverted heights
        inv_heights = np.where(mask.any(axis=0), \
                               np.argmax(mask, axis=0), \
                               self.GRID_HEIGHT)
        # Get the correct heights
        heights = self.GRID_HEIGHT - inv_heights

        # Compute the differences in pairs of adjacent columns
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)

        total_bumpiness = np.sum(diffs)
        max_bumpiness = np.max(diffs)
        return total_bumpiness, max_bumpiness

    def compute_height(self, board):
        """
        We compute the heights of each column.
        :param board: Board array
        :type board: List[List[int]]

        :return sum_height: the sum of each column's height
        :return max_height: the maximum height of all the columns
        :return min_height: the minimum height of all the columns
        """

        board = np.array(board)
        mask = board != 0

        # Get the inverted heights
        inv_heights = np.where(mask.any(axis=0), \
                               np.argmax(mask, axis=0), \
                               self.GRID_HEIGHT)
        # Get the correct heights
        heights = self.GRID_HEIGHT - inv_heights

        sum_height = np.sum(heights)
        max_height = np.max(heights)
        min_height = np.min(heights)

        return sum_height, max_height, min_height

    def get_state_size(self):
        """
        Returns number of dimensions in a given state.
        """
        return 4

    def get_next_states(self):
        """
        Get all possible next states for this piece.
        """
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        curr_pos = {'x': self.current_pos['x'], 'y': self.current_pos['y']}
        if piece_id == 0: # O piece
            num_rotations = 1
        elif piece_id == 4: # I piece
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            curr_piece, curr_pos = self.rotate_CW(curr_piece, curr_pos)
            # Loop over all possible x values the upper left corner of the
            # piece can take
            for x in range(self.GRID_WIDTH - len(curr_piece[0]) + 1):
                pos = {'x': x, \
                       'y': 0}
                # Drop the piece
                while not self.check_collision(curr_piece, pos):
                    pos['y'] += 1
                board = self.store(curr_piece, pos)
                states[(x, i+1)] = self.get_state_props(board)
        return states

    def get_current_board_state(self):
        """
        Return the current state of the board array including the currently
        active piece.

        :return board: the board with its currently placed pieces and
        the actively moving piece.
        """
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y+self.current_pos['y']][x+self.current_pos['x']] = self.piece[y][x]
        return board

    def get_game_score(self):
        """
        Return the score of the game.
        """
        return self.score

    def new_piece(self):
        """
        Generate a new piece.
        """
        self.ind = random.randrange(len(self.pieces))
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {'x': self.GRID_WIDTH//2 - len(self.piece[0])//2,
                            'y': 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        """
        Check if a collision occurred between the currently active piece and either the walls of the board or any placed piece.
        :param piece: a Piece array
        :param pos: current position of the array in the board
        :type piece: List[List[int]]
        :type pos: dict[str] = int
        """
        last_collision_row = -1
        future_y = pos['y'] + 1
        result = False

        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.GRID_HEIGHT-1:
                    result = True
                elif self.board[future_y + y][pos['x'] + x] and piece[y][x]:
                    result = True

        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos['y'] + y][pos['x'] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        # If there is overflow at the top of the board, then truncate piece and result in gameover
        if pos['y'] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0:
                if len(piece) > 1:
                    del piece[0]
                else:
                    self.gameover = True
                    return result
                last_collision_row = -1
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos['y'] + y][pos['x'] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return result

    def store(self, piece, pos):
        """
        Embed the currently active piece into the gameboard and then generate
        a new active piece.
        :param piece: a Piece array
        :param pos: current position of the array in the board
        :type piece: List[List[int]]
        :type pos: dict[str] = int
        """
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y+pos['y']][x+pos['x']]:
                    board[y+pos['y']][x+pos['x']] = piece[y][x]
        return board

    def check_cleared_rows(self):
        """
        Check for any completed rows.
        :return len(to_delete): number of rows deleted
        """
        to_delete = []
        for i, row in enumerate(self.board[::-1]):
            if 0 not in row:
                to_delete.append(len(self.board)-1-i)
        if len(to_delete) > 0:
            self.remove_row(to_delete)
        return len(to_delete)

    def remove_row(self, indices):
        """
        Remove the rows specified by a list of indices in the gameboard
        :param indices: List of row indices to be removed
        :type indices: List[int]
        """
        for i in indices[::-1]:
            del self.board[i]
            self.board = [[0 for _ in range(self.GRID_WIDTH)]] + self.board

    def play_game(self, x, num_rotations, show=True):
        """
        Given some starting x coordinate, place the piece in that column.
        :param x: x coordinate of the upper left hand corner of piece to start dropping the piece.
        :param num_rotations: the number of rotations to apply to the piece
        :param show: whether to display the gameplay
        :type x: int
        :type num_rotations: int
        :type show: Boolean
        """
        self.current_pos = {'x': x, \
                            'y': 0
                            }
        for _ in range(num_rotations):
            self.piece, self.current_pos = self.rotate_CW(self.piece, self.current_pos)
        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos['y'] += 1
            if show:
                self.show()

        self.board = self.store(self.piece, self.current_pos)
        if show:
            self.show()

        lines_cleared = self.check_cleared_rows()
        score = (1 + (lines_cleared ** 2)) * self.GRID_WIDTH
        self.score += score
        if self.gameover:
            if show:
                self.show()
            score -= 2
        else:
            self.new_piece()
        return score, self.gameover

    def show(self):
        """
        Display the current state of the board.
        """
        img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        img = np.array(img).reshape((self.GRID_HEIGHT, self.GRID_WIDTH, 3)).astype(np.uint8)
        img = img[...,::-1] # Reverse each 3-tuple in place
        img = Image.fromarray(img, "RGB")
        img = img.resize((self.GRID_WIDTH * 35, self.GRID_HEIGHT * 35))
        img = np.array(img)
        cv2.putText(img, str(self.score), (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,0))
        cv2.imshow("Tetris", np.array(img))
        cv2.waitKey(1)
