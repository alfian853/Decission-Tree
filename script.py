import pandas as pd

# def getIndexOfLowestGiniIndex(dataset,columnIndex,):
#
#     dataset.sort(key = lambda x:x[columnIndex])


file = open('dataset','r')
pd.set_option('display.max_columns', 500)
dataset = pd.read_csv('dataset')

dataTrainSize = 0.8

print(dataset)



# dataTrain = dataset[ : int( len(dataset)*dataTrainSize ) ]
# dataTest = dataset[int( len(dataset)*dataTrainSize ):]
