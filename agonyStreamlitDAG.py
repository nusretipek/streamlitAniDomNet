# Import libraries
import sys
import math
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
from HierarchiaPy import Hierarchia
from networkx.drawing.nx_agraph import graphviz_layout
import argparse

class agonyBasedDAG:
    def __init__(self, matrix: np.ndarray, timeIndex: int) -> None:
        # Parameters
        self.matrix = matrix
        self.timeIndex = timeIndex
        self.cyclesRemoved = []

        # Read Numpy file
        self.matrix = self.matrix[self.timeIndex]

        # Copy elements from strictly upper triable to NxN format
        self.copyElements()
        self.matrix = np.round(self.matrix).astype(int)

        # Get network graph
        self.graph = self.buildGraph()

    def copyElements(self):
        max_value = np.max(self.matrix[np.isfinite(self.matrix)])
        for i in range(self.matrix.shape[0]):
            for j in range(i + 1, self.matrix.shape[1]):
                if self.matrix[i, j] > self.matrix[j, i]:
                    if np.isinf(self.matrix[i, j]):
                        self.matrix[i, j] = max_value
                else:
                    self.matrix[j, i] = -self.matrix[i, j]
                    if np.isinf(self.matrix[j, i]):
                        self.matrix[j, i] = max_value
                    self.matrix[i, j] = 0

    def buildGraph(self):
        G = nx.DiGraph()
        for k in range(self.matrix.shape[0]):
            for j in range(k + 1, self.matrix.shape[1]):
                if self.matrix[k, j] == self.matrix[j, k]:
                    continue
                elif self.matrix[k, j] > self.matrix[j, k]:
                    G.add_edge(k, j, weight=self.matrix[k, j])
                else:
                    G.add_edge(j, k, weight=self.matrix[j, k])
        return G

    def getFullAdjacencyMatrix(self):
        arr = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        arr[arr > 0] = 1
        arr[arr < 1] = 0
        return arr

    def getDAG(self):
        rank = np.sum(self.matrix, axis=1) - np.sum(self.matrix, axis=0)
        for i in range(3, self.matrix.shape[0]):
            cycles = list(nx.simple_cycles(self.graph, length_bound=i))
            while len(cycles) > 0:
                agony = []
                for idz, cycle in enumerate(cycles):
                    cross_product_pairs = [(a, b) for a, b in product(cycle, repeat=2) if a != b]
                    for e1, e2 in cross_product_pairs:
                        if self.graph.get_edge_data(e1, e2) is not None:
                            agony.append((e1, e2, max(0, rank[e2] - rank[e1] + 1)))
                agony = sorted(agony, key=lambda x: x[2], reverse=True)
                self.graph.remove_edge(agony[0][0], agony[0][1])
                self.cyclesRemoved.append((agony[0][0], agony[0][1]))
                cycles = list(nx.simple_cycles(self.graph, length_bound=i))

        arr = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        arr[arr > 0] = 1
        arr[arr < 1] = 0
        return arr


class TreeGenerator:
    def __init__(self, interactionMatrix: np.ndarray, treeStructure: str = 'bottom', flatten: bool = False) -> None:
        # Init parameters
        self.treeStructure = treeStructure
        self.flatten = flatten

        # Load adjacency matrix
        self.interactionMatrix = interactionMatrix

        # Create and get dominating and dominator lists
        self.dominatorDict = {i: [] for i in range(self.interactionMatrix.shape[0])}
        self.dominatedDict = {i: [] for i in range(self.interactionMatrix.shape[0])}
        for idz in range(self.interactionMatrix.shape[0]):
            self.dominatorDict[idz] = np.where(self.interactionMatrix[idz] == 1)[0]
            self.dominatedDict[idz] = np.where(self.interactionMatrix[:, idz] == 1)[0]

        # Class variables
        self.nodeList = list(range(self.interactionMatrix.shape[0]))
        self.traversed = []
        self.tree = []

    def checkChildren(self, animalIdx: int) -> bool:
        if len(self.dominatorDict[animalIdx]) == 0:
            return True
        else:
            for child in self.dominatorDict[animalIdx]:
                if child not in self.traversed:
                    return False
            return True

    def flattenList(self) -> None:
        flat_arr = []
        for i in self.tree:
            flat_arr.extend(i)
        self.tree = flat_arr

    def _generateBottomTree(self) -> None:
        while len(self.nodeList) > 0:
            # Operation flag

            # Local variables
            keysToProcess, levelDominated, level = [], [], []
            keys = list(self.dominatorDict.keys())

            # Get keys to process
            for elementKey in keys:
                if self.checkChildren(elementKey):
                    keysToProcess.append(elementKey)

            for elementKey in keysToProcess:
                levelDominated.append(len(self.dominatedDict[elementKey]))

            # Append level
            for elementKey in np.argsort(np.array(levelDominated)):
                level.append(keysToProcess[elementKey])
                self.traversed.append(keysToProcess[elementKey])
                self.nodeList.remove(keysToProcess[elementKey])
                self.dominatorDict.pop(keysToProcess[elementKey])
                self.dominatedDict.pop(keysToProcess[elementKey])

            if len(level) > 0:
                self.tree.append(level)

        self.tree = self.tree[::-1]

    def createNetworkxGraph(self, cycles=None):
        # Create tree structure
        self._generateBottomTree()

        s = ""
        for i in self.tree:
            for j in i:
                s += (str(j) + ',')

        # Create local dominator dict
        localDominatorDict = {i: [] for i in range(self.interactionMatrix.shape[0])}
        for idx in range(self.interactionMatrix.shape[0]):
            localDominatorDict[idx] = np.where(self.interactionMatrix[idx] == 1)[0].tolist()

        localDominatorDict = {j: self.tree[i + 1] for i in range(self.tree.__len__() - 1) for j in self.tree[i]}
        localDominatedDict = {}

        for dominant, dominatedList in localDominatorDict.items():
            for d in dominatedList:
                if d not in localDominatedDict:
                    localDominatedDict[d] = []
                if self.interactionMatrix[dominant][d] == 1:
                    localDominatedDict[d].append(dominant)

        for dominated, dominantList in localDominatedDict.items():
            tempLayerLevel = None
            if len(dominantList) == 0:
                for idx, layer in enumerate(self.tree):
                    for node in layer:
                        if dominated == node:
                            tempLayerLevel = idx
                localFlag = True
                while localFlag:
                    for animal in self.tree[tempLayerLevel]:
                        if self.interactionMatrix[animal][dominated] == 1:
                            localDominatedDict[dominated].append(animal)
                            localFlag = False
                    tempLayerLevel -= 1

        # Create DiGraph
        G = nx.DiGraph()

        # Add nodes
        for idx, layer in enumerate(self.tree):
            for node in layer:
                G.add_node(node, layer=idx)

        # Add edges
        for donimated, dominantList in localDominatedDict.items():
            for animal in dominantList:
                G.add_edge(animal, donimated)

        # Positioning nodes based on layers
        pos = {}
        for i, layer in enumerate(self.tree):
            if i in []:
                layer = layer[::-1]
            for j, node in enumerate(layer):
                if i in []:
                    pos[node] = (-0.5+j*0.5, -i - np.random.uniform(0, 0.00))
                else:
                    pos[node] = (j - len(layer) / 2, -i-np.random.uniform(0,0.00))

        # Plotting the graph
        fig = plt.figure(figsize=(4, 7))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='royalblue',
                font_weight='bold', font_color='white', arrows=True, arrowstyle='->')
        return fig

    def generateTree(self) -> list:
        if self.treeStructure == 'bottom':
            self._generateBottomTree()
        else:
            raise ValueError('Tree structure not supported')

        if self.flatten:
            self.flattenList()

        return self.tree


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Dt matrix')
    parser.add_argument('-arr', type=np.ndarray, help='Dt matrix')
    return parser.parse_args()


def get_results(scoreMatrix):
    DAGobject = agonyBasedDAG(matrix=scoreMatrix, timeIndex=-1)
    DAG0 = DAGobject.getFullAdjacencyMatrix()
    DAG = DAGobject.getDAG()
    tree = TreeGenerator(DAG, treeStructure='bottom', flatten=False)
    f = tree.createNetworkxGraph(cycles=DAGobject.cyclesRemoved)
    return DAG0, DAG, f


if __name__ == '__main__':

    # Parse arguments
    args = parse_arguments()
    scoreMatrix = args.arr

    # Generate DAG - Create graph
    #DAGobject = agonyBasedDAG(scoreFileLocation=fileLocation, timeIndex=-1)
    #DAG0 = DAGobject.getFullAdjacencyMatrix()
    #DAG = DAGobject.getDAG()
    #tree = TreeGenerator(DAG, treeStructure='bottom', flatten=False)
    #f = tree.createNetworkxGraph(cycles=DAGobject.cyclesRemoved)
    #print(tree.tree)
    #print(DAGobject.cyclesRemoved)
