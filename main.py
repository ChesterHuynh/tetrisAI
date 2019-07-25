import pygame
import random
import sys
import pdb
from time import time


import piece
import tetris

points_per_line = [100, 400, 900, 2000]

# config description:
# height: number of squares in a column of the grid
# width: number of squares in a row of the grid
# gridline: width of gridlines in number of pixels
# block_size: number of pixels for each square in game grid
# speed: speed of gameplay
# move_buffer: seconds between accepting user input
# gravity_buffer: msecs between when block will be pushed down

config = {'height': 20, \
          'width': 10, \
          'gridline': 1, \
          'block_size': 35, \
          'speed': [2, 2], \
          'move_buffer': 0.05, \
          'gravity_buffer': 1000
          }

colors = {'white': (255, 255, 255), \
          'grey': (128, 128, 128), \
          'black': (0, 0, 0)
          }

def update_metrics(screen, score, lines_cleared, level, num_rows_deleted, config):
    """
    Updated the score, number of lines cleared, and level after lines have
    been cleared.
    :param screen: PyGame Surface object that we will be writing the
                   metrics onto
    :param score: Current score
    :param lines_cleared: Current number of lines cleared
    :param level: Current level
    :param num_rows_deleted: Number of lines just recently cleared
    :param config: Configurations of the game

    :type screen: PyGame Surface
    :type score: int
    :type lines_cleared: int
    :type level: int
    :type num_rows_deleted: int
    :type config: dict()

    :return score: updated score after taking into account the recently
                   cleared rows
    :return lines_cleared: updated number of lines cleared
    :return level: updated level
    """
    score += (level // 2 + 1) * points_per_line[num_rows_deleted-1]
    lines_cleared += num_rows_deleted
    level = lines_cleared // 12 # Bump level every 12 lines
    return score, lines_cleared, level

def show_metrics(screen, score, lines_cleared, level, text_size=24):
    """
    Display the current metrics on the game window
    :param screen: PyGame Surface object that we will be writing the
                   metrics onto
    :param score: Current score
    :param lines_cleared: Current number of lines cleared
    :param level: Current level
    :param text_size: Size of the text on game window

    :type screen: PyGame Surface
    :type score: int
    :type lines_cleared: int
    :type level: int
    :type text_size: int
    """

    font = pygame.font.Font(pygame.font.get_default_font(), text_size)

    # Render surfaces
    score_surface = font.render("Score: " + str(score), True, colors['white'])
    lines_cleared_surface = font.render("Lines Cleared: "+str(lines_cleared), True, colors['white'])
    level_surface = font.render("Level: " + str(level), True, colors['white'])

    # Get rect objects
    score_rect = score_surface.get_rect(centerx=(config['block_size'] + config['gridline']) * (config['width'] + 3), centery = (config['block_size'] + config['gridline']) * (1))
    lines_cleared_rect = lines_cleared_surface.get_rect(centerx=(config['block_size'] + config['gridline']) * (config['width'] + 3), centery = (config['block_size'] + config['gridline']) * (2))
    level_rect = level_surface.get_rect(centerx=(config['block_size'] + config['gridline']) * (config['width'] + 3), centery = (config['block_size'] + config['gridline']) * (3))

    screen.blit(score_surface, score_rect)
    screen.blit(lines_cleared_surface, lines_cleared_rect)
    screen.blit(level_surface, level_rect)

    pygame.display.update()

def run_game(config):
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
    board = tetris.TetrisGame(height=config['height'], width=config['width'], \
                              block_size=config['block_size'], gridline=config['gridline'], gameover=False)
    level = score = lines_cleared = 0

    show_metrics(screen, score, lines_cleared, level)
    while not board.gameover:
        config['gravity_buffer'] /= (1 + level//10)
        screen.fill(colors['black'])

        # Show HEIGHT x WIDTH grid and the active block
        board.draw(screen)
        board.piece.draw(screen)

        show_metrics(screen, score, lines_cleared, level)

        keys = pygame.key.get_pressed()
        # Add delay between function calls
        if time() - last_time > config['move_buffer']:
            if (keys[pygame.K_DOWN] or keys[pygame.K_s]):
                num_rows_deleted = board.accelerate()
                if num_rows_deleted > 0:
                    score, lines_cleared, level = update_metrics(screen, score, lines_cleared, level, num_rows_deleted, config)
            last_time = time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                board.gameover = True
            if event.type == APPLY_GRAVITY_EVENT:
                num_rows_deleted = board.accelerate()
                if num_rows_deleted > 0:
                    score, lines_cleared, level = update_metrics(screen, score, lines_cleared, level, num_rows_deleted, config)
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
