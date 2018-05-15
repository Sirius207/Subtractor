import csv
import pandas as pd

from src.dataGenerator import (getData, encodeData, splitData)
from src.model import buildModel
from src.training import train
from src.test import test

CHARS = '0123456789- '
DIGITS = 3
TOTAL_SIZE = 100000
BATCH_SIZE = 128

DATA_SIZE = {
  'TRAINING_SIZE': 30000,
  'TESTING_SIZE': 50000,
}

# unCommand to Build Corpus Data
# data = getData(TOTAL_SIZE, DIGITS)
# # Save X Corpus Data
# with open('./corpus/total_x.csv', 'w') as corpus_x:
#     for question in data[0]:
#         corpus_x.write(question + '\n')
# corpus_x.close()

# Save Y Corpus Data
# with open('./corpus/total_y.csv', 'w') as corpus_y:
#     for expect in data[1]:
#         corpus_y.write(expect + '\n')
# corpus_y.close()

# Read Corpus
data_x = list()
with open('./corpus/total_x.csv', 'r') as x:
    source = csv.reader(x, delimiter=',')
    for row in source:
        data_x.append(row[0])

data_y = list()
with open('./corpus/total_y.csv', 'r') as y:
    source = csv.reader(y, delimiter=',')
    for row in source:
        data_y.append(row[0])

data = [data_x, data_y]
encodeData = encodeData(DIGITS, data, CHARS)

diff_training_size = [10000,20000,30000,40000]

# Iterate Different Training Size
with open('./log/test_acc.csv', 'w') as output:
    output.write('model,test_acc\n')
    for training_size in diff_training_size:
        DATA_SIZE['TRAINING_SIZE'] = training_size
        TRAINING_SIZE = DATA_SIZE['TRAINING_SIZE']
        # Training data - validating data
        REAL_TRAINING_SIZE = int((TRAINING_SIZE - TRAINING_SIZE/10)/1000)

        # set training & testing data
        trainingOutputPath = './log/d' + str(DIGITS) + '/s' + str(REAL_TRAINING_SIZE)+'.csv' 
        dataSet = splitData(DATA_SIZE, encodeData)
        
        # build model & training
        model = buildModel(DIGITS, CHARS)
        training_model = train(dataSet, BATCH_SIZE, trainingOutputPath, model)
        test_acc = test(dataSet, model, CHARS)

        output.write(trainingOutputPath+',')
        output.write(str(test_acc)+'\n')




