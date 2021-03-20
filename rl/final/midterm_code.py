from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List, Mapping, Dict
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Constant, Categorical
from rl.dynamic_programming import value_iteration_result, value_iteration, almost_equal_vfs
from rl.iterate import converge, iterate
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

########## CONSTANTS ##################
SPACE = 'SPACE'
BLOCK = 'BLOCK'
GOAL = 'GOAL'


########## MDP CLASS DEFINITIONS #######################
@dataclass(frozen=True)
class MazeState:
    row: int
    col: int


MazeMoveMapping = StateActionMapping[MazeState, str]

class MazeMovementMDP(FiniteMarkovDecisionProcess[MazeState, str]):

    def __init__(
        self,
        type_dict: Mapping[Tuple[int,int],str],
        rew_fn: Mapping[Tuple[int,int],float],
    ):
        self.type_dict: Mapping[Tuple[int,int],str] = type_dict
        self.grid_size: Tuple[int,int] = (max(type_dict.keys())[0]+1,max(type_dict.keys())[1]+1)
        self.rew_fn: Mapping[Tuple[int,int],float]= rew_fn
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> MazeMoveMapping:
        d: Dict[MazeState, Dict[str, Constant[Tuple[MazeState,float]]]] = {}

        # loop through all possible states
        for i_s in range(self.grid_size[0]):
            for j_s in range(self.grid_size[1]):
                if self.type_dict[(i_s,j_s)] != SPACE: continue# do not include terminal states

                s_curr: MazeState = MazeState(i_s, j_s)
                d_s : Dict[str, Constant[Tuple[MazeState,float]]] = {}
                
                ## DOWN ##
                if i_s == self.grid_size[0]-1 or self.type_dict[i_s+1, j_s] == BLOCK:
                    d_s["DOWN"] = Constant((s_curr,self.rew_fn[(i_s,j_s)]))
                else:
                    d_s["DOWN"] = Constant((MazeState(i_s+1, j_s),self.rew_fn[(i_s+1,j_s)]))

                ## UP ##
                if i_s == 0 or self.type_dict[i_s-1, j_s] == BLOCK:
                    d_s["UP"] = Constant((s_curr,self.rew_fn[(i_s,j_s)]))
                else:
                    d_s["UP"] = Constant((MazeState(i_s-1, j_s),self.rew_fn[(i_s-1,j_s)]))

                ## RIGHT ##
                if j_s == self.grid_size[1]-1 or self.type_dict[i_s, j_s+1] == BLOCK:
                    d_s["RIGHT"] = Constant((s_curr,self.rew_fn[(i_s,j_s)]))
                else:
                    d_s["RIGHT"] = Constant((MazeState(i_s, j_s+1),self.rew_fn[(i_s,j_s+1)]))

                ## LEFT ##
                if j_s == 0 or self.type_dict[i_s, j_s-1] == BLOCK:
                    d_s["LEFT"] = Constant((s_curr,self.rew_fn[(i_s,j_s)]))
                else:
                    d_s["LEFT"] = Constant((MazeState(i_s, j_s-1),self.rew_fn[(i_s,j_s-1)]))
                d[s_curr] = d_s
        return d


########## PLOTTING TOOLS #######################
plotting_correspondance = {"NaN":0, "UP": 1, "DOWN": 2,"RIGHT": 3,"LEFT": 4}
plotting_correspondance_rev = ["N/A", "UP", "DOWN","RIGHT","LEFT"]
def discrete_matshow(data, i):
    #get discrete colormap
    cmap = plt.get_cmap('RdBu', np.max(data)-np.min(data)+1)
    # set limits .5 outside true range
    mat = plt.matshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)

    #tell the colorbar to tick at integers
    func = lambda x,pos: plotting_correspondance_rev[int(x)]
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1), format = matplotlib.ticker.FuncFormatter(func))
    ax = plt.gca();

    # Major ticks
    ax.set_xticks(np.arange(0, data.shape[0], 1))
    ax.set_yticks(np.arange(0, data.shape[1], 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[1], 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.show()


################# IMPLEMENTATION TO GIVEN MAZE DICTIONARY ########################
if __name__ == '__main__':

    maze_grid = {(0, 0): SPACE, (0, 1): BLOCK, (0, 2): SPACE, (0, 3): SPACE, (0, 4): SPACE, 
             (0, 5): SPACE, (0, 6): SPACE, (0, 7): SPACE, (1, 0): SPACE, (1, 1): BLOCK,
             (1, 2): BLOCK, (1, 3): SPACE, (1, 4): BLOCK, (1, 5): BLOCK, (1, 6): BLOCK, 
             (1, 7): BLOCK, (2, 0): SPACE, (2, 1): BLOCK, (2, 2): SPACE, (2, 3): SPACE, 
             (2, 4): SPACE, (2, 5): SPACE, (2, 6): BLOCK, (2, 7): SPACE, (3, 0): SPACE, 
             (3, 1): SPACE, (3, 2): SPACE, (3, 3): BLOCK, (3, 4): BLOCK, (3, 5): SPACE, 
             (3, 6): BLOCK, (3, 7): SPACE, (4, 0): SPACE, (4, 1): BLOCK, (4, 2): SPACE, 
             (4, 3): BLOCK, (4, 4): SPACE, (4, 5): SPACE, (4, 6): SPACE, (4, 7): SPACE, 
             (5, 0): BLOCK, (5, 1): BLOCK, (5, 2): SPACE, (5, 3): BLOCK, (5, 4): SPACE, 
             (5, 5): BLOCK, (5, 6): SPACE, (5, 7): BLOCK, (6, 0): SPACE, (6, 1): BLOCK, 
             (6, 2): BLOCK, (6, 3): BLOCK, (6, 4): SPACE, (6, 5): BLOCK, (6, 6): SPACE, 
             (6, 7): SPACE, (7, 0): SPACE, (7, 1): SPACE, (7, 2): SPACE, (7, 3): SPACE, 
             (7, 4): SPACE, (7, 5): BLOCK, (7, 6): BLOCK, (7, 7): GOAL}

    rew1: Mapping[Tuple[int,int],float] = {k:-1 for k in maze_grid.keys()}
    gamma1 = 1.0
    rew2: Mapping[Tuple[int,int],float] = {k: 1 if maze_grid[k] == GOAL else 0 for k in maze_grid.keys()}
    gamma2 = 0.5


    forms_list = [(rew1,gamma1),(rew2, gamma2)]#[(rew1, gamma1)]#
    for i, tup  in enumerate(forms_list):
        rew, gamma = tup
        maze_mdp: FiniteMarkovDecisionProcess[MazeState, str] =\
            MazeMovementMDP(
                type_dict = maze_grid,
                rew_fn = rew
            )
        print("MDP Value Iteration Optimal Policy and Number of Iterations For Formulation "+ str(i+1))
        print("--------------")
        opt_vf_vi, opt_policy_vi = value_iteration_result(maze_mdp, gamma=gamma)
        plotted_opt_pl = np.zeros(maze_mdp.grid_size)
        for s in opt_policy_vi.states():
            plotted_opt_pl[s.row,s.col] = plotting_correspondance[opt_policy_vi.act(s).sample()]
        pprint(plotted_opt_pl)
        discrete_matshow(plotted_opt_pl,i)
        print("NUMBER OF ITERATIONS", len([1 for _ in converge(value_iteration(maze_mdp, gamma=gamma), done = almost_equal_vfs)]))
    