"""Extended Zhang-Shasha tree edit distance used by PHYBench EED.

Adapted from the official PHYBench repository.
"""

from __future__ import annotations

import collections

from numpy import ones, zeros


class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or list()

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def addkid(self, node, before=False):
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def get(self, label):
        if self.label == label:
            return self
        for child in self.children:
            if label in child:
                return child.get(label)
        return None


class AnnotatedTree:
    def __init__(self, root, get_children):
        self.get_children = get_children
        self.root = root
        self.nodes = []
        self.ids = []
        self.lmds = []
        self.keyroots = None

        stack = [(root, collections.deque())]
        pstack = []
        j = 0
        while stack:
            node, ancestors = stack.pop()
            nid = j
            for child in self.get_children(node):
                child_ancestors = collections.deque(ancestors)
                child_ancestors.appendleft(nid)
                stack.append((child, child_ancestors))
            pstack.append(((node, nid), ancestors))
            j += 1

        lmds = {}
        keyroots = {}
        i = 0
        while pstack:
            (node, nid), ancestors = pstack.pop()
            self.nodes.append(node)
            self.ids.append(nid)
            if not self.get_children(node):
                lmd = i
                for anc in ancestors:
                    if anc not in lmds:
                        lmds[anc] = i
                    else:
                        break
            else:
                lmd = lmds[nid]
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1

        self.keyroots = sorted(keyroots.values())


def ext_distance(
    tree_a,
    tree_b,
    get_children,
    single_insert_cost,
    insert_cost,
    single_remove_cost,
    remove_cost,
    update_cost,
):
    """Compute the extended tree edit distance between two trees."""
    tree_a, tree_b = AnnotatedTree(tree_a, get_children), AnnotatedTree(tree_b, get_children)
    size_a = len(tree_a.nodes)
    size_b = len(tree_b.nodes)
    treedists = zeros((size_a, size_b), float)
    fd = 1000 * ones((size_a + 1, size_b + 1), float)

    def treedist(x, y):
        al = tree_a.lmds
        bl = tree_b.lmds
        an = tree_a.nodes
        bn = tree_b.nodes

        fd[al[x]][bl[y]] = 0
        for i in range(al[x], x + 1):
            node = an[i]
            fd[i + 1][bl[y]] = fd[al[i]][bl[y]] + remove_cost(node)

        for j in range(bl[y], y + 1):
            node = bn[j]
            fd[al[x]][j + 1] = fd[al[x]][bl[j]] + insert_cost(node)

        for i in range(al[x], x + 1):
            for j in range(bl[y], y + 1):
                node1 = an[i]
                node2 = bn[j]
                costs = [
                    fd[i][j + 1] + single_remove_cost(node1),
                    fd[i + 1][j] + single_insert_cost(node2),
                    fd[al[i]][j + 1] + remove_cost(node1),
                    fd[i + 1][bl[j]] + insert_cost(node2),
                ]
                min_cost = min(costs)

                if al[x] == al[i] and bl[y] == bl[j]:
                    treedists[i][j] = min(min_cost, fd[i][j] + update_cost(node1, node2))
                    fd[i + 1][j + 1] = treedists[i][j]
                else:
                    fd[i + 1][j + 1] = min(min_cost, fd[al[i]][bl[j]] + treedists[i][j])

    for x in tree_a.keyroots:
        for y in tree_b.keyroots:
            treedist(x, y)

    return treedists[-1][-1]
