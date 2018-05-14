from keras.models import Sequential
from keras import layers

RNN = layers.LSTM
HIDDEN_SIZE = 128
LAYERS = 1

def buildModel(DIGITS, chars):
    MAXLEN = DIGITS + 1 + DIGITS

    print('Build model...')

    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    model.add(layers.RepeatVector(DIGITS + 1))
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()
    return model