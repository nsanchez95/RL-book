from typing import Tuple, Sequence, Set, Mapping, Dict, Callable, Optional
from dataclasses import dataclass
from operator import itemgetter
from rl.distribution import Categorical, Choose, Constant
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import StateActionMapping
from rl.markov_decision_process import FinitePolicy
from rl.dynamic_programming import value_iteration_result, V

'''
Cell specifies (row, column) coordinate
'''
Cell = Tuple[int, int]
CellSet = Set[Cell]
Move = Tuple[int, int]
'''
WindSpec specifies a random vectical wind for each column.
Each random vertical wind is specified by a (p1, p2) pair
where p1 specifies probability of Downward Wind (could take you
one step lower in row coordinate unless prevented by a block or
boundary) and p2 specifies probability of Upward Wind (could take
you onw step higher in column coordinate unless prevented by a
block or boundary). If one bumps against a block or boundary, one
incurs a bump cost and doesn't move. The remaining probability
1- p1 - p2 corresponds to No Wind.
'''
WindSpec = Sequence[Tuple[float, float]]

possible_moves: Mapping[Move, str] = {
    (-1, 0): 'D',
    (1, 0): 'U',
    (0, -1): 'L',
    (0, 1): 'R'
}


@dataclass(frozen=True)
class WindyGrid:

    rows: int  # number of grid rows
    columns: int  # number of grid columns
    blocks: CellSet  # coordinates of block cells
    terminals: CellSet  # coordinates of goal cells
    wind: WindSpec  # spec of vertical random wind for the columns
    bump_cost: float  # cost of bumping against block or boundary

    def validate_spec(self) -> bool:
        b1 = self.rows >= 2
        b2 = self.columns >= 2
        b3 = all(0 <= r < self.rows and 0 <= c < self.columns
                 for r, c in self.blocks)
        b4 = len(self.terminals) >= 1
        b5 = all(0 <= r < self.rows and 0 <= c < self.columns and
                 (r, c) not in self.blocks for r, c in self.terminals)
        b6 = len(self.wind) == self.columns
        b7 = all(0. <= p1 <= 1. and 0. <= p2 <= 1. and p1 + p2 <= 1.
                 for p1, p2 in self.wind)
        b8 = self.bump_cost > 0.
        return all([b1, b2, b3, b4, b5, b6, b7, b8])

    def print_wind_and_bumps(self) -> None:
        for i, (d, u) in enumerate(self.wind):
            print(f"Column {i:d}: Down Prob = {d:.2f}, Up Prob = {u:.2f}")
        print(f"Bump Cost = {self.bump_cost:.2f}")
        print()

    @staticmethod
    def add_tuples(a: Cell, b: Cell) -> Cell:
        return a[0] + b[0], a[1] + b[1]

    def is_valid_state(self, cell: Cell) -> bool:
        '''
        checks if a cell is a valid state of the MDP
        '''
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns \
            and cell not in self.blocks

    def get_all_nt_states(self) -> CellSet:
        '''
        returns all the non-terminal states
        '''
        return {(i, j) for i in range(self.rows) for j in range(self.columns)
                if (i, j) not in set.union(self.blocks, self.terminals)}

    def get_actions_and_next_states(self, nt_state: Cell) \
            -> Set[Tuple[Move, Cell]]:
        '''
        given a non-terminal state, returns the set of all possible
        (action, next_state) pairs
        '''
        temp: Set[Tuple[Move, Cell]] = {(a, WindyGrid.add_tuples(nt_state, a))
                                        for a in possible_moves}
        return {(a, s) for a, s in temp if self.is_valid_state(s)}

    def get_transition_probabilities(self, nt_state: Cell) \
            -> Mapping[Move, Categorical[Tuple[Cell, float]]]:
        '''
        given a non-terminal state, return a dictionary whose
        keys are the valid actions (moves) from the given state
        and the corresponding values are the associated probabilities
        (following that move) of the (next_state, reward) pairs.
        The probabilities are determined from the wind probabilities
        of the column one is in after the move. Note that if one moves
        to a goal cell (terminal state), then one ends up in that
        goal cell with 100% probability (i.e., no wind exposure in a
        goal cell).
        '''
        d: Dict[Move, Categorical[Tuple[Cell, float]]] = {}
        for a, (r, c) in self.get_actions_and_next_states(nt_state):
            if (r, c) in self.terminals:
                d[a] = Categorical({((r, c), -1.): 1.})
            else:
                list_probs = {}

                # check if wind can blow
                has_up = (r+1 <self.rows) and ((r+1,c) not in self.blocks)
                has_down = (r > 0) and ((r-1,c) not in self.blocks)

                # probability of intended move (no wind)
                list_probs[((r,c),-1.)] = 1.-self.wind[c][0]-self.wind[c][1]

                # wind blows up no bump
                if has_up and self.wind[c][1] >0.0:
                    list_probs[((r+1,c),-1.)] = self.wind[c][1]
                
                # wind blows down no bump
                if has_down and self.wind[c][0] >0.0:
                    list_probs[((r-1,c),-1.)] = self.wind[c][0]
    
                # bump possibility!
                if not(has_up and has_down): 
                    list_probs[((r,c),-1.-self.bump_cost)] = \
                            self.wind[c][1]*(1-int(has_up))\
                                +self.wind[c][0]*(1-int(has_down))
                d[a] = Categorical(list_probs)
        return d

    def get_finite_mdp(self) -> FiniteMarkovDecisionProcess[Cell, Move]:
        '''
        returns the FiniteMarkovDecision object for this windy grid problem
        '''
        d1: StateActionMapping[Cell, Move] = \
            {s: self.get_transition_probabilities(s) for s in
             self.get_all_nt_states()}
        d2: StateActionMapping[Cell, Move] = {s: None for s in self.terminals}
        return FiniteMarkovDecisionProcess({**d1, **d2})

    def get_vi_vf_and_policy(self) -> Tuple[V[Cell], FinitePolicy[Cell, Move]]:
        '''
        Performs the Value Iteration DP algorithm returning the
        Optimal Value Function (as a V[Cell]) and the Optimal Policy
        (as a FinitePolicy[Cell, Move])
        '''
        return value_iteration_result(self.get_finite_mdp(), gamma=1.)

    @staticmethod
    def epsilon_greedy_action(
        nt_state: Cell,
        q: Mapping[Cell, Mapping[Move, float]],
        epsilon: float
    ) -> Move:
        '''
        given a non-terminal state, a Q-Value Function (in the form of a
        {state: {action: Expected Return}} dictionary) and epislon, return
        an action sampled from the probability distribution implied by an
        epsilon-greedy policy that is derived from the Q-Value Function.
        '''
        action_values: Mapping[Move, float] = q[nt_state]
        greedy_action: Move = max(action_values.items(), key=itemgetter(1))[0]
        return Categorical(
            {a: epsilon / len(action_values) +
             (1 - epsilon if a == greedy_action else 0.)
             for a in action_values}
        ).sample()

    def get_states_actions_dict(self) -> Mapping[Cell, Optional[Set[Move]]]:
        '''
        Returns a dictionary whose keys are the states and the corresponding
        values are the set of actions for the state (if the key is a
        non-terminal state) or is None if the state is a terminal state.
        '''
        d1: Mapping[Cell, Optional[Set[Move]]] = \
            {s: {a for a, _ in self.get_actions_and_next_states(s)}
             for s in self.get_all_nt_states()}
        d2: Mapping[Cell, Optional[Set[Move]]] = \
            {s: None for s in self.terminals}
        return {**d1, **d2}

    def get_sarsa_vf_and_policy(
        self,
        states_actions_dict: Mapping[Cell, Optional[Set[Move]]],
        sample_func: Callable[[Cell, Move], Tuple[Cell, float]],
        episodes: int = 10000,
        step_size: float = 0.01
    ) -> Tuple[V[Cell], FinitePolicy[Cell, Move]]:
        '''
        states_actions_dict gives us the set of possible moves from
        a non-block cell.
        sample_func is a function with two inputs: state and action,
        and with output as a sampled pair of (next_state, reward).
        '''
        q: Dict[Cell, Dict[Move, float]] = \
            {s: {a: 0. for a in actions} for s, actions in
             states_actions_dict.items() if actions is not None}
        nt_states: CellSet = {s for s in q}
        uniform_states: Choose[Cell] = Choose(nt_states)
        for episode_num in range(episodes):
            epsilon: float = 1.0 / (episode_num + 1)
            state: Cell = uniform_states.sample()
            '''
            write your code here
            update the dictionary q initialized above according
            to the SARSA algorithm's Q-Value Function updates.
            '''
            # go through a full episode
            action = self.epsilon_greedy_action(state, q,epsilon)
            while(True):
                next_state, reward = sample_func(state,action)
                
                # end while loop if end of road
                if(states_actions_dict[next_state] is None):
                    q[state][action] += step_size*(reward - q[state][action])
                    break

                next_action = self.epsilon_greedy_action(next_state, q,epsilon)
                q[state][action] += step_size*(
                    reward + q[next_state][next_action] - q[state][action])
                state = next_state
                action= next_action


        vf_dict: V[Cell] = {s: max(d.values()) for s, d in q.items()}
        policy: FinitePolicy[Cell, Move] = FinitePolicy(
            {s: Constant(max(d.items(), key=itemgetter(1))[0])
             for s, d in q.items()}
        )
        return (vf_dict, policy)

    def get_q_learning_vf_and_policy(
        self,
        states_actions_dict: Mapping[Cell, Optional[Set[Move]]],
        sample_func: Callable[[Cell, Move], Tuple[Cell, float]],
        episodes: int = 10000,
        step_size: float = 0.01,
        epsilon: float = 0.1
    ) -> Tuple[V[Cell], FinitePolicy[Cell, Move]]:
        '''
        states_actions_dict gives us the set of possible moves from
        a non-block cell.
        sample_func is a function with two inputs: state and action,
        and with output as a sampled pair of (next_state, reward).
        '''
        q: Dict[Cell, Dict[Move, float]] = \
            {s: {a: 0. for a in actions} for s, actions in
             states_actions_dict.items() if actions is not None}
        nt_states: CellSet = {s for s in q}
        uniform_states: Choose[Cell] = Choose(nt_states)
        for episode_num in range(episodes):
            state: Cell = uniform_states.sample()
            '''
            write your code here
            update the dictionary q initialized above according
            to the Q-learning algorithm's Q-Value Function updates.
            '''
           # go through a full episode
            action = self.epsilon_greedy_action(state, q,epsilon)
            while(True):
                next_state, reward = sample_func(state,action)
                
                # end while loop if end of road
                if(states_actions_dict[next_state] is None):
                    q[state][action] += step_size*(reward - q[state][action])
                    break

                next_action = self.epsilon_greedy_action(next_state, q,epsilon)
                q[state][action] += step_size*(
                    reward + max(q[next_state].values()) - q[state][action])
                state = next_state
                action= next_action
        vf_dict: V[Cell] = {s: max(d.values()) for s, d in q.items()}
        policy: FinitePolicy[Cell, Move] = FinitePolicy(
            {s: Constant(max(d.items(), key=itemgetter(1))[0])
             for s, d in q.items()}
        )
        return (vf_dict, policy)

    def print_vf_and_policy(
        self,
        vf_dict: V[Cell],
        policy: FinitePolicy[Cell, Move]
    ) -> None:
        display = "%5.2f"
        display1 = "%5d"
        vf_full_dict = {
            **{s: display % -v for s, v in vf_dict.items()},
            **{s: display % 0.0 for s in self.terminals},
            **{s: 'X' * 5 for s in self.blocks}
        }
        print("   " + " ".join([display1 % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d " % i + " ".join(vf_full_dict[(i, j)]
                                        for j in range(self.columns)))
        print()
        pol_full_dict = {
            **{s: possible_moves[policy.act(s).value]
               for s in self.get_all_nt_states()},
            **{s: 'T' for s in self.terminals},
            **{s: 'X' for s in self.blocks}
        }
        print("   " + " ".join(["%2d" % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d  " % i + "  ".join(pol_full_dict[(i, j)]
                                          for j in range(self.columns)))
        print()


if __name__ == '__main__':
    wg = WindyGrid(
        rows=5,
        columns=5,
        blocks={(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)},
        terminals={(3, 4)},
        wind=[(0., 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)],
        bump_cost=4.0
    )
    print(wg.get_finite_mdp())
    print(wg.get_actions_and_next_states((0,0)))
    valid = wg.validate_spec()
    if valid:
        wg.print_wind_and_bumps()
        vi_vf_dict, vi_policy = wg.get_vi_vf_and_policy()
        print("Value Iteration\n")
        wg.print_vf_and_policy(
            vf_dict=vi_vf_dict,
            policy=vi_policy
        )
        mdp: FiniteMarkovDecisionProcess[Cell, Move] = wg.get_finite_mdp()


        def sample_func(state: Cell, action: Move) -> Tuple[Cell, float]:
            return mdp.step(state, action).sample()

        sarsa_vf_dict, sarsa_policy = wg.get_sarsa_vf_and_policy(
            states_actions_dict=wg.get_states_actions_dict(),
            sample_func=sample_func,
            episodes=100000,
            step_size=0.01
        )
        print("SARSA\n")
        wg.print_vf_and_policy(
            vf_dict=sarsa_vf_dict,
            policy=sarsa_policy
        )

        ql_vf_dict, ql_policy = wg.get_q_learning_vf_and_policy(
            states_actions_dict=wg.get_states_actions_dict(),
            sample_func=sample_func,
            episodes=30000,
            step_size=0.01,
            epsilon=0.1
        )
        print("Q-Learning\n")
        wg.print_vf_and_policy(
            vf_dict=ql_vf_dict,
            policy=ql_policy
        )

    else:
        print("Invalid Spec of Windy Grid")
    