import pygame
import random

class Piece:
    colors = [
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
    def __init__(self, ind=random.randrange(7),x=3, y=0, block_size=35, gridline=1):
        """
        Initialize piece surface object
        :param x: current x grid coordinate of piece
        :param y: current y grid coordinate of piece
        :param block_size: size of each rectangle comprising tetrimino in pixels
        :param gridline: gridline width in pixels

        :type x, y, block_size, gridline: int
        """
        self.ind = ind
        self.piece = self.pieces[self.ind]
        self.shape = (len(self.piece), len(self.piece[0]))
        self.x = x
        self.y = y
        self.block_size = block_size
        self.gridline = gridline

    def copy(self):
        """
        Create a copy of this object to avoid shallow copies
        """
        return Piece(ind=self.ind, x=self.x, y=self.y, block_size=self.block_size, gridline=self.gridline)


    def draw(self, screen):
        """
        Draw the surfaces onto game window as Rectangles.
        :param screen: display screen which to draw the piece on
        :type screen: PyGame display
        """
        for i in range(len(self.piece)):
            for j in range(len(self.piece[i])):
                val = self.piece[i][j]
                rect = pygame.Rect((self.block_size + self.gridline) * (self.x + j), (self.block_size + self.gridline) * (self.y + i), self.block_size, self.block_size)
                pygame.draw.rect(screen, self.colors[val], rect)
