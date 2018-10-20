from unittest import TestCase

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from classifier.prune import prune


class TestPrune(TestCase):

    def setUp(self):
        input_data = np.zeros((12, 2))
        target = np.zeros((12, 1))

        for i in range(0, 12):
            input_data[i][0] = 0.1 * i
            input_data[i][1] = 1 - 0.1 * i
            target[i][0] = np.math.floor(i / 3)

        self.mytree = DecisionTreeRegressor().fit(input_data, target)
        prune(self.mytree.tree_, 0)

    def test_n_sample_reset(self):
        self.assertEqual(sum(self.mytree.tree_.n_node_samples), 12)

    def test_children_reset(self):
        self.assertEqual(sum(1 * (self.mytree.tree_.children_left != -1)), 0)
        self.assertEqual(sum(1 * (self.mytree.tree_.children_right != -1)), 0)

    pass
