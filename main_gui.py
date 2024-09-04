from tkinter import Frame, Label, CENTER, RIGHT
from random import randint
import time
import csv
import os
import time
import numpy as np
import math
from numba import jit
from main_cli import GameBoard, AI

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.grid_cells = []

        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        self.AI = AI()

        # Initialize score and create score label
        self.score = 0
        self.score_label = Label(self, text="Score: 0", font=("Verdana", 24))
        self.score_label.grid(row=0, column=GRID_LEN, sticky='nsew')

        # Load game data and create labels for highest score and tile
        self.highest_score, self.highest_tile = self.load_game_data()
        self.highest_score_label = Label(self, text=f"Highest Score: {self.highest_score}", font=("Verdana", 24))
        self.highest_score_label.grid(row=1, column=GRID_LEN, sticky='nsew')
        self.highest_tile_label = Label(self, text=f"Highest Tile: {self.highest_tile}", font=("Verdana", 24))
        self.highest_tile_label.grid(row=2, column=GRID_LEN, sticky='nsew')

        self.run_game()
        self.mainloop()

    
    def log_high_score(self):
        with open('high_scores.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.score, self.highest_tile])


    def run_game(self):
        start_time = time.perf_counter()

        while True:
            move = self.AI.get_move(self.board)
            if move is None:
                print("No valid moves left. Game over.")
                break

            self.board.move(move)
            self.update_grid_cells()
            self.add_random_tile()
            self.update_grid_cells()

            # Update score
            self.score = self.calculate_score()
            self.score_label.configure(text=f"Score: {self.score}")

            if len(self.board.get_available_moves()) == 0:
                end_time = time.perf_counter()
                game_duration = end_time - start_time
                max_tile = str(self.board.get_max_tile())
                print("GAME OVER (max tile): " + max_tile)

                # Read the last game count from the CSV file
                try:
                    with open('game_log.csv', 'r', newline='') as file:
                        last_line = list(csv.reader(file))[-1]
                        game_count = int(last_line[0]) + 1
                except (IndexError, FileNotFoundError):
                    game_count = 1

                # Write the new game count, max tile, start time, end time, and game duration to the CSV file
                with open('game_log.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    if os.stat('game_log.csv').st_size == 0:
                        writer.writerow(['Game Count', 'Max Tile', 'Start Time', 'End Time', 'Game Duration'])
                    writer.writerow([game_count, max_tile, start_time, end_time, game_duration])

                print(f"Number of games completed: {game_count}")
                break

            self.update()            

    def calculate_score(self):
        score = 0
        for row in self.board.grid:
            for cell in row:
                if cell > 2:
                    score += 2 * cell + cell
        return score
    

    def load_game_data(self):
        try:
            with open('game_data.csv', 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
                highest_score = max(data, key=lambda x: float(x[1]))[1]
                highest_tile = max(data, key=lambda x: float(x[2]))[2]
                return highest_score, highest_tile
        except FileNotFoundError:
            return 0, 0
        
    
        
    def log_game_data(self):
        with open('game_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.time(), self.score, self.highest_tile])


        
    def game_over_display(self):
        for i in range(4):
            for j in range(4):
                self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)

        self.grid_cells[1][1].configure(text="TOP",bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="4 TILES:",bg=BACKGROUND_COLOR_CELL_EMPTY)
        top_4 = list(map(int, reversed(sorted(list(self.board.grid.flatten())))))
        self.log_game_data()
        self.grid_cells[2][0].configure(text=str(top_4[0]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][1].configure(text=str(top_4[1]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][2].configure(text=str(top_4[2]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][3].configure(text=str(top_4[3]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.update()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()

        for i in range(GRID_LEN):
            grid_row = []

            for j in range(GRID_LEN):

                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.board = GameBoard()
        self.add_random_tile()
        self.add_random_tile()

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = int(self.board.grid[i][j])
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    n = new_number
                    if new_number > 2048:
                        c = 2048
                    else:
                        c = new_number

                    self.grid_cells[i][j].configure(text=str(n), bg=BACKGROUND_COLOR_DICT[c], fg=CELL_COLOR_DICT[c])
        self.update_idletasks()
        
    def add_random_tile(self):
        if randint(0,99) < 100 * 0.9:
            value = 2
        else:
            value = 4

        cells = self.board.get_available_cells()
        pos = cells[randint(0, len(cells) - 1)] if cells else None

        if pos is None:
            return None
        else:
            self.board.insert_tile(pos, value)
            return pos


def play_game():
    gamegrid = GameGrid()
    pass

def play_games_for_time(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        play_game()

# Play games for 300 seconds
play_games_for_time(300)