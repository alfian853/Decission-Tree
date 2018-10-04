import pandas as pd
import numpy as np
import math

# return from -1 to n_row + 1
def getIndexOfLowestGiniSplit(dataset,targetColumnIndex):

    rowCount = dataset.shape[0]

    #return value
    #result.type : {numeric-split,class-split}
    #numeric split using binary split
    #class split using multiway split
    result = {}
    result['type'] = None
    result['gini-split'] = None
    #split for numeric data
    result['index-split'] = None
    result['index-split-data'] = []


    if isinstance(dataset.iloc[0,targetColumnIndex],str):
        result['type'] = 'class-split'
        uniqueValue = {}
        uniqueSubsetSum = {}
        for i,data in enumerate(dataset.iloc[:,targetColumnIndex]):
            resultClass = dataset.iloc[i,dataset.columnCount-1]
            if uniqueValue.get(data) == None:
                uniqueValue[data] = {resultClass : 0}
                uniqueSubsetSum[data] = 0
            elif uniqueValue.get(data).get(resultClass) == None:
                uniqueValue[data][resultClass] = 0
            uniqueValue[data][resultClass] += 1
            uniqueSubsetSum[data]+=1

        # print(uniqueValue)
        # print(uniqueSubsetSum)

        giniSplit = math.inf

        for values in uniqueValue:
            tempGiniSpit = 1
            for value in uniqueValue[values]:
                tempGiniSpit  -= (uniqueValue[values][value]/uniqueSubsetSum[values])**2
            giniSplit = min(giniSplit,tempGiniSpit)
        # #

        result['gini-split'] = giniSplit

    else:
        result['type'] = 'numeric-split'
        dataset.sort_values(by=dataset.columns[targetColumnIndex], inplace=True)
        prefixSum = [[0 for i in range(dataset.classCount)] for j in range(dataset.shape[0]+1)]

        #prefixsum
        for j, dataClass in enumerate(dataset.classList):
            if (dataset.iloc[0,dataset.columnCount-1] == dataClass):
                prefixSum[1][j] += 1
                break

        for i,data in enumerate(dataset.iloc[1:,dataset.columnCount-1],start=1):
            i+=1
            for j,dataClass in enumerate(dataset.classList):
                if(data == dataClass):
                    prefixSum[i][j]+=1
                prefixSum[i][j]+=prefixSum[i-1][j]


        giniSplit = math.inf
        giniSplitIndex = -1
        for i in range(rowCount+1):
            #le : lower equal
            leRowSum = float(sum(prefixSum[i]))
            # g : greater
            gRowSum = rowCount - leRowSum

            #count '<=' and '>'
            leGiniIndex = 1
            gGiniIndex = 1
            for j in range(dataset.classCount):
                if leRowSum > 0:
                    leGiniIndex -= (prefixSum[i][j]/leRowSum)**2
                if gRowSum > 0:
                    greaterIJCount = prefixSum[len(prefixSum)-1][j] - prefixSum[i][j]
                    gGiniIndex -= (greaterIJCount/gRowSum)**2

            tempGiniSpit = ((leRowSum/rowCount)*leGiniIndex)  + ((gRowSum/rowCount)*gGiniIndex)
            if tempGiniSpit < giniSplit:
                giniSplit = tempGiniSpit
                giniSplitIndex = i-1
        #endfor

        print('c',giniSplit)
        print(giniSplitIndex)
        # print(dataset.iloc[[giniSplitIndex]])

        result['gini-split'] = giniSplit
        result['index-split'] = giniSplitIndex
        result['index-split-data'] = [[dataset.head(giniSplitIndex+1)],[dataset.tail(rowCount-giniSplitIndex-1)]]

    #end-else

    return result


# numeric dataset  = {'iris-dataset'}
# class dataset  = {'survey-dataset'}
pd.set_option('display.max_columns', 20)
dataset = pd.read_csv('iris-dataset')


dataset = pd.DataFrame(dataset)
dataset.columnCount = dataset.shape[1]
# print(dataset.columnCount)
dataset.classCount = dataset[dataset.columns[dataset.columnCount - 1]].nunique()
dataset.classList = dataset.iloc[:,dataset.columnCount-1].unique().tolist()
#dataset.columnCount not setted because its dynamic row
#dataset.columnCount = dataset.shape[0]
res = getIndexOfLowestGiniSplit(dataset,2)
print(res)
