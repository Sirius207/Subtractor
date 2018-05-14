REVERSE = False

from .characterTable import getCharacterTable

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def test(DATA, model, CHARS):
    print('Testing...')
    ctable = getCharacterTable(CHARS)

    test_x = DATA[2][0]
    test_y = DATA[2][1]

    right = 0
    preds = model.predict_classes(test_x, verbose=0)
    for i in range(len(preds)):
        q = ctable.decode(test_x[i])
        correct = ctable.decode(test_y[i])
        guess = ctable.decode(preds[i], calc_argmax=False)
        # print('Q', q[::-1] if REVERSE else q, end=' ')
        # print('T', correct, end=' ')
        if correct == guess:
            # print(colors.ok + '☑' + colors.close, end=' ')
            right += 1
        # else:
            # print(colors.fail + '☒' + colors.close, end=' ')
            # print(guess)
    print("MSG : Accuracy is {}".format(right / len(preds)))
    return (right / len(preds))
