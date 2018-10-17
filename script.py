import time

import pandas as pd
import numpy as np
import math
import json


class Node:
    # nodeType : {'numeric-split','class-split','leaf'}
    nodeType = None
    # numeric type node example : {'lowEqual' : nodeLeft, 'greater' : nodeRight}
    # class type node example : {'x' : nodeX, 'y' : nodeY}
    # childNodes = None
    childNodes = None

    # used for numeric split
    # for x : splitter value,  nodeLeft.value <= x < nodeRight.value
    splitterValue = -1

    # used for leaf value
    value = 'random'

    # used for node name
    name = '[result]'

    # column name or compared column
    columnName = None

    def __init__(self, nodeType='root'):
        self.nodeType = nodeType

    def addChild(self, child, childName='[result]'):
        if self.childNodes == None:
            self.childNodes = dict()
        self.childNodes[childName] = child

    def setSplitterValue(self, splitterValue):
        self.splitterValue = splitterValue

    def setLeafValue(self, value):
        self.value = value

    def setName(self, name):
        self.name = name
        # print('setchildName:' + name)

    def setColumnName(self, columnName):
        self.columnName = columnName


class DTree:
    rootNode = None
    numLeaf = 0
    numData = 0
    dataColumns = []
    dataTrainRowCount = 0
    classList = []

    def getGiniSplit(self, dataset, targetColumnIndex, classList):
        rowCount = dataset.shape[0]

        # return value
        # result.type : {numeric-split,class-split}
        # numeric split using binary split
        # class split using multiway split
        result = {}
        result['type'] = None
        result['gini-split'] = None
        # split for numeric data
        result['index-split'] = None
        result['index-split-data'] = []
        result['gini-index'] = {}

        columnCount = dataset.shape[1]

        if isinstance(dataset.iloc[0, targetColumnIndex], str):
            result['type'] = 'class-split'
            uniqueValue = {}
            uniqueSubsetSum = {}
            for i, data in enumerate(dataset.iloc[:, targetColumnIndex]):
                resultClass = dataset.iloc[i, columnCount - 1]
                if uniqueValue.get(data) is None:
                    uniqueValue[data] = {resultClass: 0}
                    uniqueSubsetSum[data] = 0
                elif uniqueValue.get(data).get(resultClass) is None:
                    uniqueValue[data][resultClass] = 0
                uniqueValue[data][resultClass] += 1
                uniqueSubsetSum[data] += 1

            # print(uniqueValue)
            # print(uniqueSubsetSum)

            giniSplit = 0
            for values in uniqueValue:
                tempGiniSpit = 1
                for value in uniqueValue[values]:
                    tempGiniSpit -= (uniqueValue[values][value] / uniqueSubsetSum[values]) ** 2
                result['gini-index'][values] = tempGiniSpit
                # print(values + " = " + str(tempGiniSpit))
                giniSplit += (uniqueSubsetSum[values] / rowCount) * tempGiniSpit
            # #
            result['gini-split'] = giniSplit

        else:
            result['type'] = 'numeric-split'
            dataset.sort_values(by=dataset.columns[targetColumnIndex], inplace=True)
            prefixSum = [[0 for i in range(len(classList))] for j in range(dataset.shape[0] + 1)]

            # prefixsum
            for j, dataClass in enumerate(classList):
                if dataset.iloc[0, columnCount - 1] == dataClass:
                    prefixSum[1][j] += 1
                    break

            for i, data in enumerate(dataset.iloc[1:, columnCount - 1], start=1):
                i += 1
                for j, dataClass in enumerate(classList):
                    if data == dataClass:
                        prefixSum[i][j] += 1
                    prefixSum[i][j] += prefixSum[i - 1][j]

            giniSplit = math.inf
            giniSplitIndex = rowCount - 1
            for i in range(rowCount + 1):
                # skip row with the same value with next row
                if i < rowCount - 1:
                    if (abs(dataset.iloc[i, targetColumnIndex] - dataset.iloc[i - 1, targetColumnIndex])) < 0.00001:
                        continue

                # le : lower equal
                leRowSum = float(sum(prefixSum[i]))
                # g : greater
                gRowSum = rowCount - leRowSum

                # count '<=' and '>'
                leGiniIndex = 1
                gGiniIndex = 1
                for j in range(len(classList)):
                    if leRowSum > 0:
                        leGiniIndex -= (prefixSum[i][j] / leRowSum) ** 2
                    if gRowSum > 0:
                        greaterIJCount = prefixSum[len(prefixSum) - 1][j] - prefixSum[i][j]
                        gGiniIndex -= (greaterIJCount / gRowSum) ** 2

                tempGiniSpit = ((leRowSum / rowCount) * leGiniIndex) + ((gRowSum / rowCount) * gGiniIndex)
                if tempGiniSpit < giniSplit:
                    giniSplit = tempGiniSpit
                    giniSplitIndex = i - 1
                    result['gini-index']['lowEqual'] = leGiniIndex
                    result['gini-index']['greater'] = gGiniIndex
            # endfor

            result['gini-split'] = giniSplit
            result['index-split'] = giniSplitIndex
            if giniSplitIndex < rowCount - 1:
                result['split-value'] = sum(dataset.iloc[giniSplitIndex:giniSplitIndex + 2, targetColumnIndex]) / 2
            else:
                result['split-value'] = dataset.iloc[giniSplitIndex, targetColumnIndex] + 0.00001

            result['index-split-data'] = [
                pd.DataFrame(dataset.iloc[:giniSplitIndex + 1].copy(), columns=dataset.columns),
                pd.DataFrame(dataset.iloc[giniSplitIndex + 1:].copy(), columns=dataset.columns)
            ]
        # end-else

        return result

    def createDecissionTree(self, dataset):
        start_t = time.time()
        self.dataColumns = dataset.columns.tolist()
        self.dataTrainRowCount = dataset.shape[0]

        columnCount = len(self.dataColumns)
        self.classList = dataset.iloc[:, columnCount - 1].unique().tolist()
        self.rootNode = self.__createDecissionTree__(dataset.copy())
        self.rootNode.setName('root')
        print('running time: ',time.time() - start_t,'second')

    # private
    def __createDecissionTree__(self, dataset):
        minGiniSplit = math.inf
        splitTarget = None
        columnCount = dataset.shape[1]

        for i in range(columnCount - 1):
            result = self.getGiniSplit(dataset, i, self.classList)
            # print('ginisplit of ' + dataset.columns[i] + ' = ' + str(result['gini-split']))
            if result['gini-split'] < minGiniSplit:
                splitTarget = result
                splitTarget['column'] = i
                minGiniSplit = result['gini-split']

        # print('chosen gini = ' + dataset.columns[splitTarget['column']])
        # print('chosen gini split val = ', splitTarget['gini-split'])
        splitTarget['column-name'] = dataset.columns[splitTarget['column']]
        # ! Spliting data process
        node = Node(splitTarget['type'])
        if splitTarget['type'] == 'class-split':
            dataSplit = {}
            categories = dataset[splitTarget['column-name']].unique().tolist()
            for category in categories:
                dataSplit[category] = dataset[dataset[splitTarget['column-name']] == category]
                dataSplit[category].pop(splitTarget['column-name'])

            for childName in categories:
                # print('traverse from ' + splitTarget['column-name'] + " = " + childName)
                if splitTarget['gini-index'][childName] - 0.005 < 0:
                    childNode = Node('leaf')
                    childNode.setLeafValue(dataSplit[childName].iloc[0, dataSplit[childName].shape[1] - 1])
                    childNode.setName(splitTarget['column-name'] + " = " + childName)
                    node.addChild(childNode, childName)
                    self.numLeaf += dataSplit[childName].shape[0]
                    print('leaf created : ', self.numLeaf, '/', self.dataTrainRowCount)
                else:
                    childNode = self.__createDecissionTree__(dataSplit[childName])
                    childNode.setName(splitTarget['column-name'] + " = " + childName)
                    node.addChild(childNode, childName)

        # endif
        elif splitTarget['type'] == 'numeric-split':
            dataSplit = {}
            dataSplit[0] = splitTarget['index-split-data'][0]
            dataSplit[1] = splitTarget['index-split-data'][1]
            node.setSplitterValue(splitTarget['split-value'])
            # print('len check',dataSplit[0].shape[0],dataSplit[1].shape[0])
            # print('split val',splitTarget['split-value'])
            # print('split column',splitTarget['column-name'])
            # print('split 1',splitTarget['split-value'])
            # print(dataSplit[0])
            # print('low equal',splitTarget['gini-index']['lowEqual'])
            # print('split 2',splitTarget['split-value'])
            # print(dataSplit[1])
            # print('greater',splitTarget['gini-index']['greater'])

            if splitTarget['gini-index']['lowEqual'] - 0.005 < 0:
                # print('x add leaf :' + dataSplit[0].iloc[0, dataSplit[0].shape[1] - 1])
                childNode = Node('leaf')
                childNode.setLeafValue(dataSplit[0].iloc[0, dataSplit[0].shape[1] - 1])
                node.addChild(childNode, 'lowEqual')
                self.numLeaf += dataSplit[0].shape[0]
                print('leaf created : ', self.numLeaf, '/', self.dataTrainRowCount)
            elif dataSplit[0].shape[0] > 0:
                childNode = self.__createDecissionTree__(dataSplit[0])
                childNode.setName(splitTarget['column-name'] + ' <= ' + str(splitTarget['split-value']))
                node.addChild(childNode, 'lowEqual')

            if splitTarget['gini-index']['greater'] - 0.005 < 0:
                # print('y add leaf :' + dataSplit[1].iloc[0, dataSplit[1].shape[1] - 1])
                childNode = Node('leaf')
                childNode.setLeafValue(dataSplit[1].iloc[0, dataSplit[1].shape[1] - 1])
                node.addChild(childNode, 'greater')
                self.numLeaf += dataSplit[1].shape[0]
                print('leaf created : ', self.numLeaf, '/', self.dataTrainRowCount)
            elif dataSplit[1].shape[0] > 0:
                childNode = self.__createDecissionTree__(dataSplit[1])
                childNode.setName(splitTarget['column-name'] + ' > ' + str(splitTarget['split-value']))
                node.addChild(childNode, 'greater')

        node.setColumnName(splitTarget['column-name'])
        return node

    def __validateDataColumn__(self, dataColumns):
        return self.dataColumns == dataColumns

    def predict(self,data):
        if not self.__validateDataColumn__(data.index.tolist()):
            raise Exception('dataset has different column from training dataset')
        return self.__predict__(data)

    # predict one row data
    def __predict__(self, data):
        node = self.rootNode
        while node.columnName is not None:
            res = data[node.columnName]

            if isinstance(res, str):
                node = node.childNodes[data[node.columnName]]
            else:
                if res <= node.splitterValue+0.000001:
                    node = node.childNodes['lowEqual']
                else:
                    node = node.childNodes['greater']

        return node.value

    def test(self, dataset):
        if not self.__validateDataColumn__(dataset.columns.tolist()):
            raise Exception('dataset has different column from training dataset')
        error = 0
        columnCount = dataset.shape[1]
        for i in range(dataset.shape[0]):
            # print(dataset.iloc[i])
            res = self.__predict__(dataset.iloc[i])
            if res != dataset.iloc[i][columnCount - 1]:
                error += 1

        print('Accuracy :',100-(error / dataset.shape[0])*100,'%')

    def exportTree(self):
        dtree = dict()
        dtree['numLeaf'] = self.numLeaf
        dtree['numData'] = self.numData
        dtree['dataColumns'] = self.dataColumns
        dtree['classList'] = self.classList
        dtree['dataTrainRowCount'] = self.dataTrainRowCount
        dtree['rootNode'] = self._exportTree(self.rootNode)

        return dtree

    def _exportTree(self, node):
        if node.__dict__['nodeType'] == 'leaf':
            return node.__dict__
        a = node.__dict__
        data = {}
        copies = node.__dict__['childNodes']
        node.__dict__['childNodes'] = dict()
        for key in zip(copies):
            node.__dict__['childNodes'][key[0]] = self._exportTree(copies[key[0]])

        return node.__dict__

    def importTree(self, tree_json):
        tree_dict = json.loads(tree_json)

        self.numLeaf = tree_dict['numLeaf']
        self.numData = tree_dict['numData']
        self.dataColumns = tree_dict['dataColumns']
        self.classList = tree_dict['classList']
        self.dataTrainRowCount = tree_dict['dataTrainRowCount']
        self.rootNode = Node()

        self.rootNode = self._importTree(self.rootNode, tree_dict['rootNode'])

    def _importTree(self, node, tree_dict):
        if tree_dict['nodeType'] == 'leaf':
            node.setLeafValue(tree_dict['value'])
            return node

        node = Node(tree_dict['nodeType'])
        node.setName(tree_dict['name'])
        node.setColumnName(tree_dict['columnName'])
        if(tree_dict.get('splitterValue')):
            node.setSplitterValue(tree_dict['splitterValue'])


        for child in tree_dict['childNodes']:
            temp = self._importTree(Node(tree_dict['childNodes'][child]['nodeType']), tree_dict['childNodes'][child])
            node.addChild(temp, child)

        return node



# numeric dataset  = {'iris-dataset'}
# class dataset  = {'survey-dataset'}
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('expand_frame_repr', False)

# dataset = pd.read_csv('tes')
# dataset = pd.read_csv('iris-dataset')
# dataset = pd.read_csv('survey-dataset')
dataset = pd.read_csv('adult.data')
# # dataset = pd.DataFrame(dataset)
dataset = pd.DataFrame(dataset)
#
#
# # Shuffling dataset. Only use when data are sorted and not large
dataset = dataset.reindex(np.random.permutation(dataset.shape[0]))
#
# # Split dataset into two sets of data, training set and validation set
split_percent = 0
split_length = math.floor(dataset.shape[0]*split_percent)
train_set = dataset.iloc[:split_length]
validation_set = dataset.iloc[split_length:]
#
# # Build Dec Tree and test it
tree = DTree()
tree.createDecissionTree(train_set)
# tree.test(validation_set)
#
# # Export built Dec Tree
tree_json = json.dumps(tree.exportTree())
f = open("decission_tree.json", "w")
f.write(tree_json)
f.close()

# Import Dec Tree from JSON file
f = open("adult_DTree.json", "r")
tree_json = f.read()
f.close()
tree = DTree()
tree.importTree(tree_json)
tree.test(validation_set)
