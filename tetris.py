# Reference: https://github.com/nuno-faria/tetris-ai/blob/master/tetris.py
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
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
    BLOCK_SIZE = 20

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
        self.score = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {'x': self.GRID_WIDTH//2 - len(self.piece[0])//2,
                            'y': 0
                            }
        self.gameover = False
        return self.get_state_props(self.board)

    def rotate_CW(self, piece):
        """
        Rotate the currently active piece clockwise
        """
        num_rows_orig = num_cols_new = len(piece)
        num_cols_orig = num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig-1)-j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_props(self, board):
        """
        Get properties of the current state of the board
        :param board: 2D list representation of board
        :type board: List[List[int]]

        :return state vector of features about board
        """
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.count_holes(board)
        total_bumpiness, max_bumpiness = self.bumpiness(board)
        sum_height, max_height, min_height = self.compute_height(board)

        return [lines_cleared, holes, total_bumpiness, sum_height]

    def count_holes(self, board):
        """
        We count the number of zeros below a non-zero entry.
        :param board: 2D list representation of board
        :type board: List[List[int]]

        :return num_holes: number of zeros in the board with nonzero values above them
        """
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.GRID_HEIGHT and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row+1:] if x == 0])
        return num_holes

    def bumpiness(self, board):
        """
        We total the differences in heights between each column of the board.
        The height of a column is the nonzero value at the highest point in the
        column.
        :param board: 2D list representation of board
        :type board: List[List[int]]

        :return total_bumpiness: sum of the differences in heights of adjacent columns
        :return max_bumpiness: max of the differences in heights of adjacent columns
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
        :return states: dictionary of actions and corresponding state properties
        """
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        curr_pos = {'x': self.current_pos['x'], 'y': self.current_pos['y']}
        if piece_id == 0: # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            # Loop over all possible x values the upper left corner of the
            # piece can take
            valid_xs = self.GRID_WIDTH - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {'x': x, \
                       'y': 0}
                # Drop the piece
                while not self.check_collision(piece, pos):
                    pos['y'] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_props(board)
            curr_piece = self.rotate_CW(curr_piece)
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
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
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
        :param pos: current position of the piece in the board
        :type piece: List[List[int]]
        :type pos: dict[str] = int

        :return True if there is a potential collision between pieces already embedded into the board and the currently active piece.
        """
        future_y = pos['y'] + 1
        result = False
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.GRID_HEIGHT-1 or self.board[future_y + y][pos['x'] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        """
        Truncate a piece that hits the game window ceiling. We set gameover to true
        once a truncation occurs (because we hit the ceiling).
        :param piece: a Piece array
        :param pos: current position of the piece in the board
        :type piece: List[List[int]]
        :type pos: dict[str] = int

        :return gameover: whether the game has ended or not
        """
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos['y'] + y][pos['x'] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        # If there is overflow at the top of the board, then truncate piece and result in gameover
        if pos['y'] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1 # Reset
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos['y'] + y][pos['x'] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        """
        Embed the currently active piece into the gameboard and then generate
        a new active piece.
        :param piece: a Piece array
        :param pos: current position of the array in the board
        :type piece: List[List[int]]
        :type pos: dict[str] = int

        :return board: updated board
        """
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y+pos['y']][x+pos['x']]:
                    board[y+pos['y']][x+pos['x']] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        """
        Check for any completed rows.
        :return len(to_delete): number of rows deleted
        :return board: the updated board
        """
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board)-1-i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        """
        Remove the rows specified by a list of indices in the gameboard
        :param indices: List of row indices to be removed
        :type indices: List[int]
        :return board: the updated board
        """
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.GRID_WIDTH)]] + board
        return board

    def play_game(self, x, num_rotations, show=True, video=None):
        """
        Given some starting x coordinate, place the piece in that column.
        :param x: x coordinate of the upper left hand corner of piece to start dropping the piece.
        :param num_rotations: the number of rotations to apply to the piece
        :param show: whether to display the gameplay
        :type x: int
        :type num_rotations: int
        :type show: Boolean

        :return score: score of the game
        :return gameover: whether or not the game ended after this action
        """
        self.current_pos = {'x': x, \
                            'y': 0
                            }
        for _ in range(num_rotations):
            self.piece = self.rotate_CW(self.piece)

        if x + len(self.piece[0]) > 10:
            pdb.set_trace()
        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos['y'] += 1
            if show:
                self.show(video)

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + ((lines_cleared ** 2)) * self.GRID_WIDTH
        self.score += score
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        return score, self.gameover

    def show(self, video=None):
        """
        Display the current state of the board.
        :param video: VideoWriter object which to write frames to record gameplay
        :type video: cv2.VideoWriter
        """
        if not self.gameover:
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.GRID_HEIGHT, self.GRID_WIDTH, 3)).astype(np.uint8)
        img = img[...,::-1] # Reverse each 3-tuple in place
        img = Image.fromarray(img, "RGB")
        img = img.resize((self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))
        img = np.array(img)

        winname = "Tetris"
        cv2.putText(img, str(self.score), (50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,0,0))
        if video:
            video.write(img)
        cv2.imshow(winname, np.array(img))
        cv2.waitKey(1)
