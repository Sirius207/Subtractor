from keras.models import Sequential
from keras import layers

from six.moves import range

from src.dataGenerator import (getEncodeData, splitData)
from src.model import buildModel
from src.training import train
from src.test import test

CHARS = '0123456789- '
DIGITS = 3
TOTAL_SIZE = 80000
BATCH_SIZE = 128

DATA_SIZE = {
  'TRAINING_SIZE': 30000,
  'TESTING_SIZE': 50000,
}

encodeData = getEncodeData(TOTAL_SIZE, DIGITS, CHARS)

test_DIGITS = [3,4]
test_training_size = [10000]

with open('test_acc.csv', 'w') as output:
    output.write('model,test_acc\n')

    for training_size in test_training_size:
        DATA_SIZE['TRAINING_SIZE'] = training_size
        TRAINING_SIZE = DATA_SIZE['TRAINING_SIZE']
        REAL_TRAINING_SIZE = int((TRAINING_SIZE - TRAINING_SIZE/10)/1000)

        trainingOutputPath = './log/d' + str(DIGITS) + '/s' + str(REAL_TRAINING_SIZE)+'.csv' 
        dataSet = splitData(DATA_SIZE, encodeData)
        
        model = buildModel(DIGITS, CHARS)
        training_model = train(dataSet, BATCH_SIZE, trainingOutputPath, model)
        test_acc = test(dataSet, model, CHARS)

        output.write(trainingOutputPath+',')
        output.write(str(test_acc)+'\n')




