import pygame
import piece
import random
import copy
import pdb

class TetrisGame:
    colors = [
        (128, 128, 128),
        (255, 0,   0  ),
        (0,   150, 0  ),
        (0,   0,   255),
        (255, 120, 0  ),
        (255, 255, 0  ),
        (180, 0,   255),
        (0,   220, 220)
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

    def __init__(self, height=16, width=10, block_size=35, gridline=1, gameover=False):
        """
        Initialize a tetris board
        :param height: how many blocks tall the tetris board should be
        :param width: how many blocks wide the tetris board should be
        :param block_size: how many pixels each block is comprised of
        :param gridline: width of gridlines in pixels
        :param gameover: gameover status

        :type height: int
        :type width: int
        :type block_size: int
        :type gridline: int
        :type gameover: boolean
        """
        # Initialize board of zeros
        self.height = height
        self.width = width
        self.block_size = block_size
        self.gridline = gridline
        self.gameover = gameover
        self.piece = piece.Piece(x=3, y=0, block_size=self.block_size, gridline=self.gridline)
        self.board = []
        for y in range(height):
            row = [0] * width
            self.board.append(row)

    def __str__(self):
        """
        The current state of the tetris board as a string
        """
        out = ""
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                out += str(self.board[i][j]) + "  "
            out += "\n"
        return out

    def new_piece(self):
        """
        Generate a random new piece
        """
        ind = random.randrange(7)
        new = piece.Piece(ind, x=3, y=0, block_size=self.block_size, gridline=self.gridline)
        new.x = int(self.width/2 - new.shape[1]/2)
        if self.check_collision(new):
            self.gameover = True
        self.piece = new

    def translate(self, dx):
        """
        Translates tetris piece by dx grid squares
        :param dx: 1 or -1 to move piece right or left, respectively.
        :type dx: int
        """
        new_x = self.piece.x + dx
        if new_x < 0:
            new_x = 0
        if new_x > self.width - self.piece.shape[1]:
            # self.piece.x = self.width - self.piece.shape[1]
            new_x = self.width - self.piece.shape[1]
        potential_new = self.piece.copy()
        potential_new.shape = self.piece.shape
        potential_new.x = new_x
        if not self.check_collision(potential_new):
            self.piece.x = new_x

    def accelerate(self):
        """
        Accelerates tetris piece down by one grid square.

        :return num_rows_deleted: number of rows that were deleted after this
                                  acceleration call.
        """
        num_rows_deleted = 0
        self.piece.y += 1
        if self.piece.y > self.height - self.piece.shape[0]:
            self.piece.y = self.height - self.piece.shape[0]
            self.store()
            num_rows_deleted = self.check_row_cleared()

        if self.check_collision(self.piece):
            self.join_pieces()
            num_rows_deleted = self.check_row_cleared()

        return num_rows_deleted

    def rotate_CW(self):
        """
        Rotate a tetris piece clockwise.
        """
        # # Had problems with x and y coordinates
        num_rows_orig = num_cols_new = self.piece.shape[0]
        num_cols_orig = num_rows_new = self.piece.shape[1]
        rotated_array = []

        if self.piece.x + self.piece.shape[0] > self.width:
            x_offset = self.piece.x + self.piece.shape[0] - self.width
            self.piece.x -= x_offset
        if self.piece.y + self.piece.shape[1] > self.height:
            y_offset = self.piece.y + self.piece.shape[1] - self.height
            self.piece.y -= y_offset

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = self.piece.piece[(num_rows_orig-1)-j][i]
            rotated_array.append(new_row)
        rotated_piece = self.piece.copy()
        rotated_piece.piece = rotated_array
        if not self.check_collision(rotated_piece):
            self.piece.shape = (num_rows_new, num_cols_new)
            self.piece.piece = rotated_piece.piece


    def check_collision(self, piece):
        """
        :param piece: piece to check collisions for
        :type piece: Piece
        :return True if the piece is hitting a wall or another piece,
                else False
        """
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                try:
                    if self.board[piece.y + y][piece.x + x] and piece.piece[y][x]:
                        return True # We hit another tetrimino
                except IndexError:
                    return True # We hit out of bounds/a wall
        return False

    def check_row_cleared(self):
        """
        Check for any completed rows.
        :return number of rows deleted
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
            self.board = [[0 for _ in range(self.width)]] + self.board

    def store(self):
        """
        Embed the currently active piece into the gameboard and then generate
        a new active piece.
        """
        for y in range(self.piece.shape[0]):
            for x in range(self.piece.shape[1]):
                if self.piece.piece[y][x] != 0:
                    self.board[self.piece.y + y][self.piece.x + x] = self.piece.piece[y][x]
        self.new_piece()

    def join_pieces(self):
        """
        Embed the currently active piece into the gameboard when near an
        occupied space of the gameboard, then generate a new active piece.
        """
        for y in range(self.piece.shape[0]):
            for x in range(self.piece.shape[1]):
                self.board[self.piece.y + y - 1][self.piece.x + x] += self.piece.piece[y][x]
        self.new_piece()

    def draw(self, screen):
        """
        Displays the tetris board onto the actual screen.
        :param screen: PyGame Surface object to draw the board onto
        :type screen: Surface
        """
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                rect = pygame.Rect((self.block_size + self.gridline) * x, (self.block_size + self.gridline) * y, self.block_size, self.block_size)
                pygame.draw.rect(screen, self.colors[self.board[y][x]], rect)
