import numpy as np
from .characterTable import getCharacterTable

def getData(TOTAL_SIZE, DIGITS):
    REVERSE = False
    MAXLEN = DIGITS + 1 + DIGITS

    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TOTAL_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}-{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a - b)
        ans += ' ' * (DIGITS + 1 - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))
    print(questions[:5], expected[:5])
    return [questions, expected]


def encodeData(DIGITS, DATA, CHARS):
    MAXLEN = DIGITS + 1 + DIGITS

    questions = DATA[0]
    expected = DATA[1]

    ctable = getCharacterTable(CHARS)
    ctable.indices_char

    print('Vectorization...')
    x = np.zeros((len(questions), MAXLEN, len(CHARS)), dtype=np.bool)
    y = np.zeros((len(expected), DIGITS + 1, len(CHARS)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    return [x,y]
  

def getEncodeData(TOTAL_SIZE, DIGITS, CHARS):
    data = getData(TOTAL_SIZE, DIGITS)
    encode_data = encodeData(DIGITS, data, CHARS)
    return encode_data

def splitData(DATA_SIZE, encode_data):
    TRAINING_SIZE = DATA_SIZE['TRAINING_SIZE']
    TESTING_SIZE = DATA_SIZE['TESTING_SIZE']

    x = encode_data[0]
    y = encode_data[1]

    # train_test_split
    train_x = x[:TRAINING_SIZE]
    train_y = y[:TRAINING_SIZE]
    test_x = x[-TESTING_SIZE:]
    test_y = y[-TESTING_SIZE:]

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    # print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])

    return [[x_train,y_train],[x_val,y_val],[test_x,test_y]]