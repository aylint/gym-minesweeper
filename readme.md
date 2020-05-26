# Minesweeper Gym Environment

A standardized openAI gym environment implementing Minesweeper game

![](https://cardgames.io/minesweeper/images/minesweeper-logo.png)

Minesweeper is a single player puzzle game. 

In this implementation, you have an NxN board with M mines. 
Each cell on the board has an integer value assigned; from "-2" (unknown) to "9".  
Non-negative values indicate the number of neighboring cells containing mines (-1)
If a cell has 0 value (no mines as neighbors), the neighboring cells are automatically opened. 

The standardized gym functions are implemented. 

Have fun playing or teaching your code how to play. 


There are two implementations with a slight difference in the action spaces. 

**MinesweeperEnv**: MultiDiscreet action space (as a 2D matrix)

**MinesweeperDiscreetEnv**: Discreet action space (as an array)


The `step` function returns a list of valid actions (i.e. playable cells) in `info` field. 


# Installation

```
cd gym-minesweeper/
pip install -e .
```