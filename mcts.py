from typing import List
from kaggle_environments.envs.hungry_geese.hungry_geese import (
    Observation,
    Configuration,
    Action,
    row_col,
)
from kaggle_environments import evaluate, make, utils, Environment
import numpy as np


Actions = [Action.SOUTH, Action.WEST, Action.NORTH, Action.EAST]
ActionOp = [1, 0, -1, 0, 1]

N = 11
M = 7
num_opponent_action_space = 3


def oppositeAction(action_name: str) -> str:
    if action_name == Action.SOUTH.name:
        return Action.NORTH.name
    if action_name == Action.NORTH.name:
        return Action.SOUTH.name
    if action_name == Action.EAST.name:
        return Action.WEST.name
    if action_name == Action.WEST.name:
        return Action.EAST.name


class QNode:
    def __init__(self, action: Action) -> None:
        self.wins = 0
        self.sims = 0
        self.action = action
        self.next = []


def getActionCandidates(
    geese: List[List[int]], last_actions: List[str]
) -> List[List[str]]:
    """Get action candidates for all geese"""
    feasible_actions = []
    for goose, last_action in zip(geese, last_actions):
        if len(goose) == 0:
            feasible_actions.append([])
            continue
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
                and Actions[i].name == oppositeAction(last_action)
            ):
                continue

            actions.append(Actions[i].name)
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

    def _isDead(
        self,
    ) -> bool:
        return self.env.done or self.env.state[self.player_ind]["status"] == "INACTIVE"

    def getActionCandidates(self) -> List[List[str]]:
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
            sample = []
            for i, acts in enumerate(actions):
                if i == player_ind:
                    continue
                sample.append(np.random.choice(acts))
            result_set.add(tuple(sample))
        return [list(r) for r in result_set]


class MCTS:
    def __init__(self, state, steps, player_ind) -> None:
        """Monte Carlo Tree Search"""
        root_env = make("hungry_geese", state=state, steps=steps, debug=True)
        last_actions = [agent.action for agent in state]
        self.root = Node(root_env, player_ind, last_actions)
        self.player_ind = player_ind

    def select(self) -> List[Node]:
        """Select a path from root"""
        tmp = self.root
        path = [self.root]
        while tmp.next:
            tmp = np.random.choice(tmp.next, size=1, replace=False)
            path.append(tmp)
        return path

    def expansion(self, path: List[Node]) -> List[Node]:
        """Expand node for feasible moves"""
        node = path[-1]
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
                child = Node(node.env.clone(), self.player_ind, oppoent_act)
                child.env.step(oppoent_act)
                Qchild.next.append(child)

        sample_qnode = np.random.choice(node.next)
        sample_node = np.random.choice(sample_qnode.next)
        path.append(sample_qnode)
        path.append(sample_node)
        return path

    def sim(self, node: Node, steps: int) -> bool:
        sim_env = node.env.clone()
        last_actions = node.last_actions
        for _ in range(steps):
            geese = sim_env.state[self.player_ind].observation.geese
            actions_candidates = getActionCandidates(geese, last_actions)
            actions = [
                act_list[np.random.randint(0, len(act_list))]
                for act_list in actions_candidates
            ]
            last_actions = actions
            sim_env.step(actions)
            if sim_env.done or sim_env.state[self.player_ind].status == "DONE":
                break
        rewards = [state.reward for state in sim_env.state]
        return rewards.index(max(rewards)) == self.player_ind

    def backprop(self, path: List[Node], result: bool) -> None:
        for node in path:
            if result:
                node.wins += 1
            node.sims += 1
