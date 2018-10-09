import pandas as pd
import numpy as np
import math
import json

def getGiniSplit(dataset, targetColumnIndex, classList=[]):
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

        giniSplit = 1
        giniSplitIndex = rowCount-1
        for i in range(rowCount + 1):
            # skip row with the same value with next row
            if i < rowCount-1:
                if (abs(dataset.iloc[i,targetColumnIndex] - dataset.iloc[i-1,targetColumnIndex]) ) < 0.00001:
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

        # print('c', giniSplit)
        # print(giniSplitIndex)
        # print(dataset.iloc[[giniSplitIndex]])

        result['gini-split'] = giniSplit
        result['index-split'] = giniSplitIndex
        if giniSplitIndex<rowCount-1:
            result['split-value'] = sum(dataset.iloc[giniSplitIndex:giniSplitIndex+2, targetColumnIndex])/2
        else:
            result['split-value'] = dataset.iloc[giniSplitIndex, targetColumnIndex] + 0.00001
        # frame1 = pd.DataFrame(dataset.iloc[:giniSplitIndex + 1].copy(), columns=dataset.columns)
        # frame1.append(,columns=dataset.columns)
        # frame1.set
        result['index-split-data'] = [
            pd.DataFrame(dataset.iloc[:giniSplitIndex + 1].copy(), columns=dataset.columns),
            pd.DataFrame(dataset.iloc[giniSplitIndex+1:].copy(), columns=dataset.columns)
        ]
        # print('return ')
        # print(frame1)
    # end-else

    return result


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

    def __init__(self, nodeType):
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


# class = unique count of result class
# used for numeric data
def createDecissionTree(dataset, classList=[]):
    minGiniSplit = math.inf
    splitTarget = None
    columnCount = dataset.shape[1]

    for i in range(columnCount - 1):
        result = getGiniSplit(dataset, i, classList)
        # print('ginisplit of ' + dataset.columns[i] + ' = ' + str(result['gini-split']))
        if result['gini-split'] < minGiniSplit:
            splitTarget = result
            splitTarget['column'] = i
            minGiniSplit = result['gini-split']


    # print('chosen gini = ' + dataset.columns[splitTarget['column']])
    splitTarget['column-name'] = dataset.columns[splitTarget['column']]
    # ! Spliting data process
    dataSplit = {}
    node = Node(splitTarget['type'])

    if splitTarget['type'] == 'class-split':
        poppedColumn = dataset.pop(splitTarget['column-name'])
        for i, splitterClass in enumerate(poppedColumn):

            if dataSplit.get(splitterClass) is None:
                dataSplit[splitterClass] = pd.DataFrame(columns=dataset.columns)

            dataSplit[splitterClass] = dataSplit[splitterClass].append(dataset.iloc[i, :].copy())
        # endfor

        for childName in dataSplit:
            # print('traverse from ' + splitTarget['column-name'] + " = " + childName)
            if splitTarget['gini-index'][childName] - 0.005 < 0:
                # print('x====result :' + dataSplit[childName].iloc[0, dataSplit[childName].shape[1] - 1])
                childNode = Node('leaf')
                childNode.setLeafValue(dataSplit[childName].iloc[0, dataSplit[childName].shape[1] - 1])
                childNode.setName(splitTarget['column-name'] + " = " + childName)
                node.addChild(childNode, childName)
            else:
                childNode = createDecissionTree(dataSplit[childName])
                childNode.setName(splitTarget['column-name'] + " = " + childName)
                node.addChild(childNode, childName)

    # endif
    elif splitTarget['type'] == 'numeric-split':
        dataSplit[0] = splitTarget['index-split-data'][0]
        dataSplit[1] = splitTarget['index-split-data'][1]
        node.setSplitterValue(splitTarget['split-value'])
        print('len check',dataSplit[0].shape[0],dataSplit[1].shape[0])
        print('split 1',splitTarget['split-value'])
        print(dataSplit[0])
        print(splitTarget['gini-index']['lowEqual'])
        print('split 2',splitTarget['split-value'])
        print(dataSplit[1])
        print(splitTarget['gini-index']['greater'])

        if splitTarget['gini-index']['lowEqual'] - 0.005 < 0:
            # print('x add leaf :' + dataSplit[0].iloc[0, dataSplit[0].shape[1] - 1])
            childNode = Node('leaf')
            childNode.setLeafValue(dataSplit[0].iloc[0, dataSplit[0].shape[1] - 1])
            node.addChild(childNode,'lowEqual')
        else:
            childNode = createDecissionTree(dataSplit[0], classList)
            childNode.setName(splitTarget['column-name'] + ' <= ' + str(splitTarget['split-value']))
            node.addChild(childNode, 'lowEqual')

        if splitTarget['gini-index']['greater'] - 0.005 < 0:
            # print('y add leaf :' + dataSplit[1].iloc[0, dataSplit[1].shape[1] - 1])
            childNode = Node('leaf')
            childNode.setLeafValue(dataSplit[1].iloc[0, dataSplit[1].shape[1] - 1])
            node.addChild(childNode,'greater')
        else:
            childNode = createDecissionTree(dataSplit[1], classList)
            childNode.setName(splitTarget['column-name'] + ' > ' + str(splitTarget['split-value']))
            node.addChild(childNode,'greater')
    # print(node.splitterValue)
    # for child in node.childNodes:
    #     print(child,node.childNodes[child].name,node.childNodes[child].columnName)
    # input()
    # ! End of Spliting data process

    node.setColumnName(splitTarget['column-name'])
    return node


def predict(decissionTree, data):
    node = decissionTree
    while node.columnName is not None:
        # print(node.name, node.columnName)
        # print(data[node.columnName])
        # print(node.nodeType)
        res = data[node.columnName]
        if isinstance(res, str):
            node = node.childNodes[data[node.columnName]]
        else:
            if res <= node.splitterValue:
                node = node.childNodes['lowEqual']
            else:
                node = node.childNodes['greater']
        # input()
    return node.value


# numeric dataset  = {'iris-dataset'}
# class dataset  = {'survey-dataset'}
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

# dataset = pd.read_csv('tes')
dataset = pd.read_csv('iris-dataset')
#dataset = pd.read_csv('survey-dataset')
dataset = pd.DataFrame(dataset)

columnCount = dataset.shape[1]
classCount = dataset[dataset.columns[columnCount - 1]].nunique()
classList = dataset.iloc[:, columnCount - 1].unique().tolist()

dataset = dataset.reindex(np.random.permutation(dataset.shape[0]))
train_set = dataset.iloc[0:100]
validation_set = dataset.iloc[100:]

#print(validation_set.shape)


# tree = createDecissionTree(dataset.copy(), classList)
tree = createDecissionTree(train_set.copy(), classList)
tree.setName('root')
# error = 0
# for i in range(validation_set.shape[0]):
#     # print(dataset.iloc[i])
#     res = predict(tree, validation_set.iloc[i])
#     if res != validation_set.iloc[i][columnCount - 1]:
#         error += 1
#         print(res,'vs',validation_set.iloc[i][columnCount-1])
#         #input()
# #
# print(error / validation_set.shape[0])

def iter(node):
    if node.__dict__['nodeType'] == 'leaf':
        return node.__dict__
    a = node.__dict__
    #print(a)
    data = {}
    copies = node.__dict__['childNodes']
    node.__dict__['childNodes'] = dict()
    for key in zip(copies):
        node.__dict__['childNodes'][key] = iter(copies[key[0]])

    return node.__dict__

    # print(iter(a['childNodes']['lowEqual']))
    # print(iter(a['childNodes']['greater']))

print(iter(tree))
