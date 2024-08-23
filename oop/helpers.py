import math
from constants import *

def calc_grid_coordinates(index):
    row = index // GRID_WIDTH
    col = index % GRID_WIDTH
    # print(f"calc grid coordinates: index: {index}, Row: {row}, Col: {col}")
    return row, col

def calc_raw_coordinates(index):
    return calc_raw_coordinates(*calc_grid_coordinates(index))

def calc_raw_coordinates(row, col):
    return GRID_POSITIONS[row][col]