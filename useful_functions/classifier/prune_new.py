import copy
import numpy as np
import sys

from sklearn.tree import _tree


class TreePruner:
    def __init__(self, grown_tree):
        self.trees = [grown_tree]

        num_nodes = self.trees[0].tree_.capacity
        self.parents = [-1] * num_nodes
        self.gks = sys.maxsize * np.ones(num_nodes)

    def run(self):
        self._determine_parents(self.trees[0].tree_, 0, 0)
        self.parents[0] = -1
        self._determine_gks(self.trees[0].tree_)

        num_nodes = self.trees[0].tree_.capacity
        k = 1

        while num_nodes > 1:
            self.trees.append(copy.deepcopy(self.trees[k - 1]))

            # determine alpha
            alpha_idx = self.gks.argmin()

            # prune tree in alpha_idx
            self.prune(self.trees[k].tree_, alpha_idx)

            # set gk at alpha_idx to maxsize and update all parent nodes
            self.gks[alpha_idx] = sys.maxsize
            self._update_gks(self.trees[k].tree_, alpha_idx)

            # update num_nodes
            num_nodes = sum(1 * (self.trees[k].tree_.n_node_samples != 0))
            k += 1

    def prune(self, inner_tree, index):
        """
        The given tree object is modified to resemble a version pruned at index
        @:param inner_tree: sk-learn tree object
        @:param index: the index of the node at which inner_tree is pruned
        """

        # wenn es 'children' gibt besuche ich die 'children'
        if inner_tree.children_left[index] != _tree.TREE_LEAF:
            self.prune(inner_tree, inner_tree.children_left[index])
            self.prune(inner_tree, inner_tree.children_right[index])

            # set node to leaf
            idx_left = inner_tree.children_left[index]
            idx_right = inner_tree.children_right[index]

            inner_tree.children_left[index] = _tree.TREE_LEAF
            inner_tree.children_right[index] = _tree.TREE_LEAF

            inner_tree.n_node_samples[idx_left] = 0
            inner_tree.n_node_samples[idx_right] = 0

            inner_tree.impurity[idx_left] = 0
            inner_tree.impurity[idx_right] = 0

            self.gks[idx_left] = sys.maxsize
            self.gks[idx_right] = sys.maxsize


        else:
            # wenn es keine 'children' gibt kann ich prunen
            inner_tree.n_node_samples[index] = 0
            self.gks[index] = sys.maxsize


    def _calc_gk(self, tree, node_idx):
        node_impurity = tree.n_node_samples[node_idx] * tree.impurity[node_idx]
        subtree_impurity, subtree_leafs = self._calc_impurity(tree, node_idx)

        return (node_impurity - subtree_impurity) / (subtree_leafs - 1.)


    def _determine_gks(self, tree):
        """
        Given a regression tree, the relevant penalty scalars gk are determined for pruning. Every
        inner node of the tree is visit to evaluate the penalty scalar gk that would make pruning in each node reasonable.
        The gks are stored in self.gks
        @:param tree: sk-learn tree object
        """

        # traverse all inner nodes in to find min_gk
        for node_idx in range(tree.node_count):
            # if node is a leaf node, skip node
            if tree.children_left[node_idx] == _tree.TREE_LEAF:
                continue

            # inner node
            self.gks[node_idx] = self._calc_gk(tree, node_idx)
        return


    def _update_gks(self, tree, index):

        while index != 0:
            parent_index = self.parents[index]
            self.gks[parent_index] = self._calc_gk(tree, parent_index)
            index = parent_index


    def _calc_impurity(self, tree, index):
        """
        Calc_impurity is a recursive function for calculating the absolute impurity of any subtree.
        The absolute impurity is calculated by the impurity of every leaf-node scaled with the number of samples per node.
        @:param tree: sk-learn tree object
        @:param index: the index of the root node of the subtree
        @:returns impurity and leaf count of subtree
        """

        # print("index: ", index, " impurity: ", d_tree.tree_.n_node_samples[index] * tree.impurity[index] / 10000000)
        # wenn es 'children' besuche die 'children'
        if tree.children_left[index] != _tree.TREE_LEAF:
            impurity_left, leafs_left = self._calc_impurity(tree, tree.children_left[index])
            impurity_right, leafs_right = self._calc_impurity(tree, tree.children_right[index])

            return impurity_left + impurity_right, leafs_left + leafs_right

        # wenn es keine 'children' gibt bin ich ein leaf Knoten
        else:
            # print("index: ", index, " cost: ", d_tree.tree_.n_node_samples[index] * tree.impurity[index]/10000000)
            return tree.n_node_samples[index] * tree.impurity[index], 1

    def _determine_parents(self, tree, parent_index, index):
        self.parents[index] = parent_index

        # wenn es 'children' besuche die 'children'
        if tree.children_left[index] != _tree.TREE_LEAF:
            self._determine_parents(tree, index, tree.children_left[index])
            self._determine_parents(tree, index, tree.children_right[index])

