# GridworldEnv
## Christopher Kang | Under William Agnew

Implements a basic Gridworld with cherries / bombs for use in testing basic RL agents. 

## I/O

The Gridworld has a number of methods that provide flexibility in state representation. The primary one (distance + contacts) has the following format:

  * The matrix is n x n x 3, where n = the number of objects, including the agent 
  * The first two slices, or the __distance representation__, are lower triangular
    * The 0th slice is of the X Dimension, while the 1st slice is of the Y dimension
    * First, the numbers on the diagonals represent the reward value of an object, with the exception of the agent, whose value is its coordinates
    * The row represents the target object, while the column represents the origin
    * The agent receives the 0th column, while the first object receives the first column, and so on.
    * _E.g._: The number in row 1 column 0 represents the distance from the agent to the 1st object
  * The final slice is the __contact representation__
    * This is a boolean representation of whether an object is directly touching another object
    * This matrix is again lower triangular
    * Once an agent has consumed an object, it is always considered to be contacting 
