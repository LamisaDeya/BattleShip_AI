# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 05:47:46 2024

@author: ASUS
"""


# pip install pygame
import pygame
import sys
import BattleShip_Engine

#button_label=0
if len(sys.argv) > 1:
    button_label = sys.argv[1]
else:
    button_label = "default"

print(f"Button pressed: {button_label}")

# initialize pygame
pygame.init()
# initialize font
pygame.font.init()

# window caption
pygame.display.set_caption("Battleship")
myfont = pygame.font.SysFont("fresansttf", 50)
label_font = pygame.font.SysFont("fresansttf", 30)

# global variables
SQ_SIZE = 30  # each cell
H_MARGIN = SQ_SIZE * 4  # margin between grid
V_MARGIN = SQ_SIZE  # margin between grid
WIDTH = SQ_SIZE * 10 * 2 + H_MARGIN
HEIGHT = SQ_SIZE * 10 * 2 + V_MARGIN
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
INDENT = 10
HUMAN1 = True
HUMAN2 = False
if(button_label=="1" or button_label=="2" or button_label=="3"):
    HUMAN1=True
elif (button_label=="4" or button_label=="5" or button_label=="6"):
    HUMAN1=False
GREY = (40, 50, 60)  # Colors
WHITE = (255, 250, 250)  # Colors
GREEN = (50, 200, 150)
GREEN2 = (50, 255, 150)  # Colors
RED = (250, 50, 100)
BLUE = (50, 150, 200)
ORANGE = (250, 140, 20)
COLORS = {"U": GREY, "M": BLUE, "H": ORANGE, "S": RED}
COLORS2 = {"U": GREY, "M": GREY, "H": ORANGE, "S": RED}

# Game statistics
human_hits = 0
computer_hits = 0
human_sinks = 0
computer_sinks = 0

# Drag-and-drop variables
dragging_ship = None
drag_offset_x = 0
drag_offset_y = 0

# function to draw grid
def draw_grid(player, left=0, top=0, search=False):  # top-left specified
    for i in range(100):  # 100 cells
        # x, y location of cell (col, row)
        x = left + i % 10 * SQ_SIZE  # 10 cols thakbe
        y = top + i // 10 * SQ_SIZE  # 10 rows thakbe

        # after location fixing, draw cell
        square = pygame.Rect(x, y, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(SCREEN, WHITE, square, width=3)
        if search:
            x += SQ_SIZE // 2  # center define korlam
            y += SQ_SIZE // 2
            pygame.draw.circle(SCREEN, COLORS[player.search[i]], (x, y), radius=SQ_SIZE // 4)


def draw_grid_human(player, left=0, top=0, search=False):  # top-left specified
    for i in range(100):  # 100 cells
        # x, y location of cell (col, row)
        x = left + i % 10 * SQ_SIZE  # 10 cols thakbe
        y = top + i // 10 * SQ_SIZE  # 10 rows thakbe

        # after location fixing, draw cell
        square = pygame.Rect(x, y, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(SCREEN, WHITE, square, width=3)
        if search:
            x += SQ_SIZE // 2  # center define korlam
            y += SQ_SIZE // 2
            pygame.draw.circle(SCREEN, COLORS2[player.search[i]], (x, y), radius=SQ_SIZE // 4)


# function to draw ships onto the position grids
def draw_ships(player, left=0, top=0):
    for ship in player.ships:
        x = left + ship.col * SQ_SIZE + INDENT  # 10 cols thakbe
        y = top + ship.row * SQ_SIZE + INDENT  # 10 rows thakbe
        if ship.orientation == "h":
            width = ship.size * SQ_SIZE - 2 * INDENT
            height = SQ_SIZE - 2 * INDENT
        else:
            height = ship.size * SQ_SIZE - 2 * INDENT
            width = SQ_SIZE - 2 * INDENT
        rectangle = pygame.Rect(x, y, width, height)
        pygame.draw.rect(SCREEN, GREEN, rectangle, border_radius=15)
        # indent jaate ship full cell occupy na kore


def draw_initial_ships(ships):
    for ship in ships:
        x, y, width, height = ship["rect"]
        rectangle = pygame.Rect(x, y, width, height)
        pygame.draw.rect(SCREEN, GREEN, rectangle, border_radius=15)


def is_overlap(new_ship, placed_ships):
    new_rect = pygame.Rect(new_ship["rect"])
    for ship in placed_ships:
        if new_rect.colliderect(pygame.Rect(ship["rect"])):
            return True
    return False


def is_within_bounds(ship, grid_left, grid_top):
    x, y, width, height = ship["rect"]
    return (grid_left <= x <= grid_left + 10 * SQ_SIZE - width) and (grid_top <= y <= grid_top + 10 * SQ_SIZE - height)


p1 = BattleShip_Engine.Player()
p2 = BattleShip_Engine.Player()

game = BattleShip_Engine.Game(HUMAN1, HUMAN2, p1, p2)
player1 = game.player1
player2 = game.player2

# List of ships for initial placement
initial_ships = [
    {"rect": [WIDTH // 2 - 48, 0, 5 * SQ_SIZE - 2 * INDENT, SQ_SIZE - 2 * INDENT], "size": 5, "orientation": "h"},
    {"rect": [WIDTH // 2 - 48, 100, SQ_SIZE - 2 * INDENT, 4 * SQ_SIZE - 2 * INDENT], "size": 4, "orientation": "v"},
    {"rect": [WIDTH // 2 - 48, 200, 3 * SQ_SIZE - 2 * INDENT, SQ_SIZE - 2 * INDENT], "size": 3, "orientation": "h"},
    {"rect": [WIDTH // 2 - 48, 300, SQ_SIZE - 2 * INDENT, 3 * SQ_SIZE - 2 * INDENT], "size": 3, "orientation": "v"},
    {"rect": [WIDTH // 2 - 48, 400, 2 * SQ_SIZE - 2 * INDENT, SQ_SIZE - 2 * INDENT], "size": 2, "orientation": "h"}
]


# pygame loop
animating = True
pausing = False
placing_ships = True
placed_ships = []

while animating:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            animating = False

        if placing_ships or dragging_ship:
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                for ship in initial_ships:
                    rect = pygame.Rect(ship["rect"])
                    if rect.collidepoint(x, y):
                        dragging_ship = ship
                        drag_offset_x = rect.x - x
                        drag_offset_y = rect.y - y
                        break

            if event.type == pygame.MOUSEBUTTONUP:
                if dragging_ship:
                    x, y = pygame.mouse.get_pos()
                    dragging_ship["rect"][0] = x + drag_offset_x
                    dragging_ship["rect"][1] = y + drag_offset_y
                    # Check for overlap
                    if is_overlap(dragging_ship, placed_ships) or not is_within_bounds(dragging_ship, 0, (HEIGHT - V_MARGIN) // 2 + V_MARGIN):
                        dragging_ship["rect"][0] = WIDTH // 2 - 48  # Reset to initial position
                        dragging_ship["rect"][1] = 0 + initial_ships.index(dragging_ship) * 100  # Reset to initial position
                    else:
                        if dragging_ship not in placed_ships:
                            placed_ships.append(dragging_ship)
                    dragging_ship = None

            if event.type == pygame.MOUSEMOTION:
                if dragging_ship:
                    x, y = pygame.mouse.get_pos()
                    dragging_ship["rect"][0] = x + drag_offset_x
                    dragging_ship["rect"][1] = y + drag_offset_y

            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and not dragging_ship:
                placing_ships = False
                # Convert initial ship placements to Ship objects for the player
                player1.ships = []  # Clear existing ships before adding new ones
                for ship in placed_ships:
                    col, row = ship["rect"][0] // SQ_SIZE, (ship["rect"][1] - (HEIGHT - V_MARGIN) // 2 - V_MARGIN) // SQ_SIZE
                    if ship["orientation"] == "h":
                        if col + ship["size"] > 10:
                            col = 10 - ship["size"]
                    else:
                        if row + ship["size"] > 10:
                            row = 10 - ship["size"]
                    new_ship = BattleShip_Engine.Ship(size=ship["size"], row=row, col=col, orientation=ship["orientation"])
                    player1.ships.append(new_ship)
                player1.update_indexes()

        else:
            if event.type == pygame.MOUSEBUTTONDOWN and not game.over:
                x, y = pygame.mouse.get_pos()
                if game.player1_turn and x < SQ_SIZE * 10 and y < SQ_SIZE * 10:
                    row = y // SQ_SIZE
                    col = x // SQ_SIZE
                    index = row * 10 + col
                    game.make_move(index)
                    if game.player1.search[index] == 'H':
                        human_hits += 1
                    elif game.player1.search[index] == 'S':
                        human_sinks += 1
                        human_hits += 1

                elif not game.player1_turn and x > WIDTH - SQ_SIZE * 10 and y > SQ_SIZE * 10 + V_MARGIN:
                    row = (y - SQ_SIZE * 10 - V_MARGIN) // SQ_SIZE
                    col = (x - SQ_SIZE * 10 - H_MARGIN) // SQ_SIZE
                    index = row * 10 + col
                    game.make_move(index)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    animating = False
                if event.key == pygame.K_SPACE:
                    pausing = not pausing

    if not pausing:
        SCREEN.fill(GREY)
        draw_grid_human(player1, search=True)
        draw_grid(player2, search=True, left=(WIDTH - H_MARGIN) // 2 + H_MARGIN, top=(HEIGHT - V_MARGIN) // 2 + V_MARGIN)
        draw_grid(player2, top=(HEIGHT - V_MARGIN) // 2 + V_MARGIN)

        if placing_ships or dragging_ship:
            draw_initial_ships(initial_ships)
        else:
            draw_ships(player1, left=0, top=(HEIGHT - V_MARGIN) // 2 + V_MARGIN)

        # Draw labels for grids
        your_own_grid_label = label_font.render("Your Own Ship Placement Grid", True, WHITE)
        #your_attacking_grid_label = label_font.render("Your Attacking Grid", True, WHITE)
        computer_attacking_grid_label = label_font.render("Computer's Attacking Grid", True, WHITE)

        # Labels' positions
        #SCREEN.blit(your_attacking_grid_label, (SQ_SIZE * 5 - your_attacking_grid_label.get_width() // 2, 0))
        SCREEN.blit(your_own_grid_label, (SQ_SIZE * 5 - your_own_grid_label.get_width() // 2, (HEIGHT - V_MARGIN) // 2 + V_MARGIN  -25))
        SCREEN.blit(computer_attacking_grid_label, (WIDTH // 2 +75, (HEIGHT - V_MARGIN) // 2 + V_MARGIN  -25))


        if not game.over and game.computer_turn:
            if not game.player1_turn:
                #num = game.basic_ai()
                # num = game.basic_ai()
                # num = game.random_ai()
                #num = game.basic_ai_with_fuzzy()
                '''
                num = game.minmax_ai()
                if (button_label=="4"):
                    num=game.basic_ai()
                elif (button_label=="5"):
                    num=game.basic_ai()  
                elif (button_label=="6"):
                    num=game.basic_ai_with_fuzzy()   
                elif (button_label=="1"):
                    num=game.basic_ai()
                elif (button_label=="3"):
                    num=game.minmax_ai()  
                elif (button_label=="2"):
                    num=game.basic_ai_with_fuzzy()   
                '''
                # Determine AI move based on button_label
                if button_label == "4":
                    num = game.basic_ai()
                elif button_label == "5":
                    num = game.basic_ai()
                elif button_label == "6":
                    num = game.basic_ai_with_fuzzy()
                elif button_label == "2":
                    num = game.basic_ai()
                elif button_label == "1":
                    num = game.minmax_ai()
                elif button_label == "3":
                    num = game.basic_ai_with_fuzzy()
                else:
                    num = game.minmax_ai()  # Default to minmax_ai if label is unknown
               
                if num == 1:
                    computer_hits += 1
                elif num == 2:
                    computer_hits += 1
                    computer_sinks += 1
            else:
                num=game.minmax_ai()
                
                
                if (button_label=="4"):
                    num=game.basic_ai_with_fuzzy()
                elif (button_label=="5"):
                    num=game.minmax_ai()  
                elif (button_label=="6"):
                    num=game.minmax_ai()
                '''    
                elif (button_label=="1"):
                    num=game.basic_ai()
                elif (button_label=="3"):
                    num=game.minmax_ai()  
                elif (button_label=="2"):
                    num=game.basic_ai_with_fuzzy()   
                '''

        if game.over:
            if game.result ==2:
                text = "Computer  wins!"
            else:
                text = "Human wins!"
            
            textbox = myfont.render(text, False, GREY, WHITE)
            SCREEN.blit(textbox, (WIDTH // 2 - 132, HEIGHT // 2 - 19))

        hits_text = myfont.render(f"Human Hits: {human_hits}", False, WHITE)
        computer_hits_text = myfont.render(f"Comp Hits: {computer_hits}", False, WHITE)
        sinks_text = myfont.render(f"Human Sinks: {human_sinks}", False, WHITE)
        computer_sinks_text = myfont.render(f"Comp Sinks: {computer_sinks}", False, WHITE)

        SCREEN.blit(hits_text, (WIDTH // 2 + 90, 40))
        SCREEN.blit(computer_hits_text, (WIDTH // 2 + 90, 90))
        SCREEN.blit(sinks_text, (WIDTH // 2 + 90, 140))
        SCREEN.blit(computer_sinks_text, (WIDTH // 2 + 90, 190))

        pygame.time.wait(100)
        pygame.display.flip()

pygame.quit()
