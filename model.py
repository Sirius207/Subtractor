import csv
from keras.models import load_model
from src.dataGenerator import (getData, encodeData, splitData)
from src.test import test

DIGITS = 3
CHARS = '0123456789- '
DATA_SIZE = {
  'TRAINING_SIZE': 45000,
  'TESTING_SIZE': 50000,
}

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

# Build Testing Data
data = [data_x, data_y]
encodeData = encodeData(DIGITS, data, CHARS)
dataSet = splitData(DATA_SIZE, encodeData)

# Load Model & Testing
model = load_model('digit3_epoch200_size45000.h5')
test_acc = test(dataSet, model, CHARS)