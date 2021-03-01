import unittest
from mcts import (
    MCTS,
    Node,
    oppositeAction,
    sampleOpponentActions,
    num_opponent_action_space,
)
from kaggle_environments import evaluate, make, utils, Environment
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class TestActions(unittest.TestCase):
    def test(self):
        actions = [
            [Action.NORTH, Action.SOUTH],
            [Action.NORTH, Action.WEST],
            [Action.NORTH, Action.SOUTH],
        ]

        sample_actions = sampleOpponentActions(actions, 0)
        self.assertEqual(len(sample_actions[0]), 2)
        self.assertEqual(len(sample_actions), num_opponent_action_space)
        for acts in sample_actions:
            for i, act in enumerate(acts):
                self.assertTrue(act in actions[i + 1])


class TestNode(unittest.TestCase):
    def test(self):
        env = make("hungry_geese", debug=True)
        trainer = env.train([None, "greedy"])
        trainer.step(Action.EAST.name)

        last_actions = [agent.action for agent in env.state]
        node = Node(env, 0, last_actions)

        actions = node.getActionCandidates()

        for acts, last_act in zip(actions, last_actions):
            oppo_act = oppositeAction(last_act)
            for act in acts:
                self.assertFalse(act == oppo_act)


class TestSelection(unittest.TestCase):
    def test(self):
        env = make("hungry_geese", debug=True)
        trainer = env.train([None, "greedy", "greedy", "greedy"])
        state = trainer.step(Action.EAST.name)
        simulator = MCTS(env.state, env.steps, 0)

        path = simulator.select()
        self.assertEqual(len(path), 1)

        path = simulator.expansion(path)
        self.assertEqual(len(path), 3)

        result = simulator.sim(path[-1], 10)

        simulator.backprop(path, result)

        for node in path:
            self.assertEqual(node.sims, 1)
            self.assertEqual(node.wins, 1 if result else 0)


if __name__ == "__main__":
    unittest.main()
