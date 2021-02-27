from typing import List
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation,
    Configuration,
    Action,
    row_col,
)
from kaggle_environments import evaluate, make, utils, Environment
import numpy as np
from numpy.core.fromnumeric import size


Actions = [Action.SOUTH, Action.WEST, Action.NORTH, Action.EAST]
ActionOp = [1, 0, -1, 0, 1]

N = 11
M = 7
num_opponent_action_space = 3


class QNode:
    def __init__(self, action: Action) -> None:
        self.wins = 0
        self.sims = 0
        self.action = action
        self.next = []


def getActionCandidates(
    geese: List[List[int]], last_actions: List[Action]
) -> List[List[Action]]:
    """Get action candidates for all geese"""
    feasible_actions = []
    for goose, last_action in zip(geese, last_actions):
        head_x, head_y = row_col(goose[0], N)
        actions = []
        for i in range(4):
            next_x, next_y = (
                head_x + ActionOp[i],
                head_y + ActionOp[i + 1],
            )
            next_x = (next_x + M) % M
            next_y = (next_y + N) % N

            if len(goose) > 1 and next_x * N + next_y in goose[0:-1]:
                continue
            if (
                len(goose) <= 2
                and last_action
                and Actions[i].name == last_action.opposite().name
            ):
                continue

            actions.append(Actions[i])
        feasible_actions.append(actions)

    return feasible_actions


class Node:
    def __init__(
        self, env: Environment, player_ind, last_actions: List[Action]
    ) -> None:
        self.env = env
        self.player_ind = player_ind
        self.wins = 0
        self.sims = 0
        self.next = []
        self.last_actions = last_actions

        self.is_dead = self._isDead()

    def _isDead(self,) -> bool:
        return self.env.done or self.env.state[self.player_ind]["status"] == "INACTIVE"

    def getActionCandidates(self,) -> List[List[Action]]:
        """Get action candidates for all geese"""
        geese = self.env.state[self.player_ind].observation.geese
        return getActionCandidates(geese, self.last_actions)


def sampleOpponentActions(actions: List[List[Action]], player_ind: int) -> List[List]:
    num_all_action_combinator = np.prod([len(row) for row in actions])

    if num_all_action_combinator <= num_opponent_action_space:
        result = []
        for a in actions[0]:
            for b in actions[1]:
                for c in actions[2]:
                    result.append([a, b, c])
        return result
    else:
        result_set = set()
        while len(result_set) < num_opponent_action_space:
            a = np.random.choice(actions[0])
            b = np.random.choice(actions[1])
            c = np.random.choice(actions[2])
            result_set.add((a, b, c))
        return [list(r) for r in result_set]


class MCTS:
    def __init__(self, state, steps, player_ind) -> None:
        root_env = make("Hungry_geese", state=state, steps=steps, debug=True)
        self.root = Node(root_env)
        self.player_ind = player_ind

    def select(self):
        """Select a path from root"""
        tmp = self.root
        path = [self.root]
        while tmp.next:
            tmp = np.random.choice(tmp.next, size=1, replace=False)
            path.append(tmp)
        return path

    def expansion(self, node: Node):
        """Expand node for feasible moves"""
        if node.is_dead:
            return

        actions = node.getActionCandidates()
        player_actions = actions[self.player_ind]
        for act in player_actions:
            Qchild = QNode(act)
            node.next.append(Qchild)
            opponent_actions = sampleOpponentActions(actions, self.player_ind)
            for oppoent_act in opponent_actions:
                oppoent_act.insert(self.player_ind, act)
                child = Node(node.env, self.player_ind, act)
                child.env.step(oppoent_act)
                Qchild.next.append(child)

        sample_qnode = np.random.choice(node.next)
        return np.random.choice(sample_qnode.next)

    def sim(self, node: Node, steps: int):
        sim_env = node.env.clone()
        last_actions = node.last_actions.copy()
        for _ in range(steps):
            geese = sim_env.state[self.player_ind].observation.geese
            actions_candidates = getActionCandidates(geese, last_actions)
            actions = [
                act_list[np.random.randint(0, len(act_list))]
                for act_list in actions_candidates
            ]
            last_actions = actions
            sim_env.step(actions)
            if sim_env.done:
                break
        rewards = [state.reward for state in sim_env.state]
        return rewards.index(max(rewards)) == self.player_ind

    def backprop(self):
        pass
