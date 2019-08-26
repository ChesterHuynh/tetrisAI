import pygame
import random
import sys
from time import time

import tetrisinteractive

colors = {'white': (255, 255, 255), \
          'grey': (128, 128, 128), \
          'black': (0, 0, 0)
          }

config = {'height': 20, # height: number of squares in a column of the grid
          'width': 10, # width: number of squares in a row of the grid
          'gridline': 0, # gridline: width of gridlines in number of pixels
          'block_size': 35, # block_size: number of pixels for each square in game grid
          'speed': [2, 2], # speed: speed of gameplay
          'move_buffer': 0.05, # move_buffer: seconds between accepting user input
          'gravity_buffer': 1000#  gravity_buffer: msecs between when block will be pushed down
          }

def run_game(config):
    """
    Runs the full tetris game until gameover status is True. Gameover is only
    achieved when a new block no longer fits on a screen.

    :param config: Configurations for the tetris game display
    :type config: dict
    """
    pygame.init()

    # Game window initialization
    window_width = (config['block_size'] + config['gridline']) * \
                    config['width'] + 6 * (config['block_size'] + \
                    config['gridline'])
    window_height = (config['block_size'] + config['gridline']) * \
                     config['height']
    window_size = window_width, window_height
    screen = pygame.display.set_mode(window_size)
    screen.fill(colors['black'])
    pygame.display.set_caption("TETRIS")

    # Set up gravity
    APPLY_GRAVITY_EVENT = pygame.USEREVENT+1 # event to trigger gravity event
    last_time = time()
    pygame.time.set_timer(APPLY_GRAVITY_EVENT, config['gravity_buffer'])

    # Initialize pertinant game variables
    board = tetrisinteractive.TetrisGame(height=config['height'], width=config['width'], \
                              block_size=config['block_size'], gridline=config['gridline'], gameover=False)

    board.show_metrics(screen, config)
    while not board.gameover:
        config['gravity_buffer'] /= (1 + board.level//10)
        screen.fill(colors['white'])

        # Show HEIGHT x WIDTH grid and the active block
        board.draw(screen)
        board.piece.draw(screen)

        board.show_metrics(screen, config)

        keys = pygame.key.get_pressed()
        # Add delay between function calls
        if time() - last_time > config['move_buffer']:
            if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                num_rows_deleted = board.accelerate()
                if num_rows_deleted > 0:
                    board.update_metrics(screen, num_rows_deleted, config)
            last_time = time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                board.gameover = True
            if event.type == APPLY_GRAVITY_EVENT:
                num_rows_deleted = board.accelerate()
                if num_rows_deleted > 0:
                    board.update_metrics(screen, num_rows_deleted, config)
            if event.type == pygame.KEYDOWN:
                # Quit game
                if event.key in [pygame.K_ESCAPE]:
                    board.gameover = True
                if event.key in [pygame.K_UP, pygame.K_w]:
                    board.rotate_CW()
                if event.key in [pygame.K_LEFT, pygame.K_a]:
                    board.translate(-1)
                if event.key in [pygame.K_RIGHT, pygame.K_d]:
                    board.translate(1)
        pygame.display.update()

if __name__ == "__main__":
    run_game(config)
