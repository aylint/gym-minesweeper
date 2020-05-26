from gym.envs.registration import register
from minesweeper.minesweeper import *


register(
        id='Minesweeper-v0',
        entry_point='minesweeper.minesweeper:MinesweeperEnv',
        )


register(
        id='MinesweeperDiscreet-v0',
        entry_point='minesweeper.minesweeper:MinesweeperDiscreetEnv',
)


