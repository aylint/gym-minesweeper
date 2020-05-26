import sys
from six import StringIO
from random import randint

import numpy as np
import gym
from gym import spaces

# default : easy board
BOARD_SIZE = 10
NUM_MINES = 9

# cell values, non-negatives indicate number of neighboring mines
MINE = -1
CLOSED = -2


def board2str(board, end='\n'):
    """
    Format a board as a string

    Parameters
    ----
    board : np.array
    end : str

    Returns
    ----
    s : str
    """
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += end
    return s[:-len(end)]


def is_new_move(my_board, x, y):
    """ return true if this is not an already clicked place"""
    return my_board[x, y] == CLOSED


def is_valid(x, y):
    """ returns if the coordinate is valid"""
    return (x >= 0) & (x < BOARD_SIZE) & (y >= 0) & (y < BOARD_SIZE)


def is_win(my_board):
    """ return if the game is won """
    return np.count_nonzero(my_board == CLOSED) == NUM_MINES


def is_mine(board, x, y):
    """return if the coordinate has a mine or not"""
    return board[x, y] == MINE


def place_mines(board_size, num_mines):
    """generate a board, place mines randomly"""
    mines_placed = 0
    board = np.zeros((board_size, board_size), dtype=int)
    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size)
        x = int(rnd / board_size)
        y = int(rnd % board_size)
        if is_valid(x, y):
            if not is_mine(board, x, y):
                board[x, y] = MINE
                mines_placed += 1
    return board


class MinesweeperEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, board_size=BOARD_SIZE, num_mines=NUM_MINES):
        """
        Create a minesweeper game.

        Parameters
        ----
        board_size: int     shape of the board
            - int: the same as (int, int)
        num_mines: int   num mines on board
        """

        self.board_size = board_size
        self.num_mines = num_mines
        self.board = place_mines(board_size, num_mines)
        self.my_board = np.ones((board_size, board_size), dtype=int) * CLOSED
        self.valid_actions = np.ones((self.board_size, self.board_size), dtype=np.bool)

        self.observation_space = spaces.Box(low=-2, high=9,
                                            shape=(self.board_size, self.board_size), dtype=np.int)
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size])

    def count_neighbour_mines(self, x, y):
        """return number of mines in neighbour cells given an x-y coordinate

            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if is_valid(_x, _y):
                    if is_mine(self.board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, my_board, x, y):
        """return number of mines in neighbour cells given an x-y coordinate

            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if is_valid(_x, _y):
                    if is_new_move(my_board, _x, _y):
                        my_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if my_board[_x, _y] == 0:
                            my_board = self.open_neighbour_cells(my_board, _x, _y)
        return my_board

    def get_next_state(self, state, x, y):
        """
        Get the next state.

        Parameters
        ----
        state : (np.array)   visible board
        x : int    location
        y : int    location

        Returns
        ----
        next_state : (np.array)    next visible board
        game_over : (bool) true if game over

        """
        my_board = state
        game_over = False
        if is_mine(self.board, x, y):
            my_board[x, y] = MINE
            game_over = True
        else:
            my_board[x, y] = self.count_neighbour_mines(x, y)
            if my_board[x, y] == 0:
                my_board = self.open_neighbour_cells(my_board, x, y)
        self.my_board = my_board
        return my_board, game_over

    def reset(self):
        """
        Reset a new game episode. See gym.Env.reset()

        Returns
        ----
        next_state : (np.array, int)    next board
        """
        self.board = place_mines(self.board_size, self.num_mines)
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.valid_actions = np.ones((self.board_size, self.board_size), dtype=np.bool)
        return self.my_board

    def step(self, action):
        """
        See gym.Env.step().

        Parameters
        ----
        action : np.array    location

        Returns
        ----
        next_state : (np.array)    next board
        reward : float        the reward for action
        done : bool           whether the game end or not
        info : {}
        """
        state = self.my_board
        x = int(round(action[0]))
        y = int(round(action[1]))

        # test valid action
        if bool(self.valid_actions[x, y]) is False:
            raise Exception("Invalid action was selected! Action Filter: {}, "
                            "action taken: {}".format(self.valid_actions, action))

        next_state, reward, done, info = self.next_step(state, x, y)
        self.my_board = next_state
        self.valid_actions = (next_state == CLOSED)
        info['valid_actions'] = (next_state == CLOSED)
        return next_state, reward, done, info

    def next_step(self, state, x, y):
        """
        Get the next observation, reward, done, and info.

        Parameters
        ----
        state : (np.array)    visible board
        x : int    location
        y : int    location

        Returns
        ----
        next_state : (np.array)    next visible board
        reward : float               the reward
        done : bool           whether the game end or not
        info : {}
        """
        my_board = state
        if not is_new_move(my_board, x, y):
            return my_board, -1, False, {}
        while True:
            state, game_over = self.get_next_state(my_board, x, y)
            if not game_over:
                if is_win(state):
                    return state, 1000, True, {}
                else:
                    return state, 5, False, {}
            else:
                return state, -100, True, {}

    def render(self, mode='human'):
        """
        See gym.Env.render().
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = board2str(self.board)
        outfile.write(s)
        if mode != 'human':
            return outfile


class MinesweeperDiscreetEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, board_size=BOARD_SIZE, num_mines=NUM_MINES):
        """
        Create a minesweeper game.

        Parameters
        ----
        board_size: int     shape of the board
            - int: the same as (int, int)
        num_mines: int   num mines on board
        """

        self.board_size = board_size
        self.num_mines = num_mines
        self.board = place_mines(board_size, num_mines)
        self.my_board = np.ones((board_size, board_size), dtype=int) * CLOSED
        self.num_actions = 0

        self.observation_space = spaces.Box(low=-2, high=9,
                                            shape=(self.board_size, self.board_size), dtype=np.int)
        self.action_space = spaces.Discrete(self.board_size*self.board_size)
        self.valid_actions = np.ones((self.board_size * self.board_size), dtype=np.bool)

    def count_neighbour_mines(self, x, y):
        """return number of mines in neighbour cells given an x-y coordinate

            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if is_valid(_x, _y):
                    if is_mine(self.board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, my_board, x, y):
        """return number of mines in neighbour cells given an x-y coordinate

            Cell -->Current Cell(row, col)
            N -->  North(row - 1, col)
            S -->  South(row + 1, col)
            E -->  East(row, col + 1)
            W -->  West(row, col - 1)
            N.E --> North - East(row - 1, col + 1)
            N.W --> North - West(row - 1, col - 1)
            S.E --> South - East(row + 1, col + 1)
            S.W --> South - West(row + 1, col - 1)
        """
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if is_valid(_x, _y):
                    if is_new_move(my_board, _x, _y):
                        my_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if my_board[_x, _y] == 0:
                            my_board = self.open_neighbour_cells(my_board, _x, _y)
        return my_board

    def get_next_state(self, state, x, y):
        """
        Get the next state.

        Parameters
        ----
        state : (np.array)   visible board
        x : int    location
        y : int    location

        Returns
        ----
        next_state : (np.array)    next visible board
        game_over : (bool) true if game over

        """
        my_board = state
        game_over = False
        if is_mine(self.board, x, y):
            my_board[x, y] = MINE
            game_over = True
        else:
            my_board[x, y] = self.count_neighbour_mines(x, y)
            if my_board[x, y] == 0:
                my_board = self.open_neighbour_cells(my_board, x, y)
        self.my_board = my_board
        return my_board, game_over

    def reset(self):
        """
        Reset a new game episode. See gym.Env.reset()

        Returns
        ----
        next_state : (np.array, int)    next board
        """
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.board = place_mines(self.board_size, self.num_mines)
        self.num_actions = 0
        self.valid_actions = np.ones((self.board_size * self.board_size), dtype=bool)

        return self.my_board

    def step(self, action):
        """
        See gym.Env.step().

        Parameters
        ----
        action : np.array    location

        Returns
        ----
        next_state : (np.array)    next board
        reward : float        the reward for action
        done : bool           whether the game end or not
        info : {}             {'valid_actions': valid_actions} - a binary vector,
                                where false cells' values are already known to observer
        """
        state = self.my_board
        x = int(action / self.board_size)
        y = int(action % self.board_size)

        # test valid action - uncomment this part to test your action filter if needed
        # if bool(self.valid_actions[action]) is False:
        #    raise Exception("Invalid action was selected! Action Filter: {}, "
        #                    "action taken: {}".format(self.valid_actions, action))

        next_state, reward, done, info = self.next_step(state, x, y)
        self.my_board = next_state
        self.num_actions += 1
        self.valid_actions = (next_state.flatten() == CLOSED)
        info['valid_actions'] = self.valid_actions
        info['num_actions'] = self.num_actions
        return next_state, reward, done, info

    def next_step(self, state, x, y):
        """
        Get the next observation, reward, done, and info.

        Parameters
        ----
        state : (np.array)    visible board
        x : int    location
        y : int    location

        Returns
        ----
        next_state : (np.array)    next visible board
        reward : float               the reward
        done : bool           whether the game end or not
        info : {}
        """
        my_board = state
        if not is_new_move(my_board, x, y):
            return my_board, 0, False, {}
        while True:
            state, game_over = self.get_next_state(my_board, x, y)
            if not game_over:
                if is_win(state):
                    return state, 1000, True, {}
                else:
                    return state, 0, False, {}
            else:
                return state, -100, True, {}

    def render(self, mode='human'):
        """
        See gym.Env.render().
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = board2str(self.my_board)
        outfile.write(s)
        if mode != 'human':
            return outfile
