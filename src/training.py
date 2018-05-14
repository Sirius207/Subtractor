def train(DATA, BATCH_SIZE, trainingOutputPath, model):
    
    x_train = DATA[0][0]
    y_train = DATA[0][1]
    x_val = DATA[1][0]
    y_val = DATA[1][1]

    training_log = list()
    with open(trainingOutputPath, 'w') as output:
        output.write('loss,acc,val_loss,val_acc\n')
        for iteration in range(1):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            history_callback = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=1,
                    validation_data=(x_val, y_val))
            training_log.append(history_callback.history)

            output.write(str(history_callback.history['loss'][0]) + ',')
            output.write(str(history_callback.history['acc'][0]) + ',')
            output.write(str(history_callback.history['val_loss'][0]) + ',')
            output.write(str(history_callback.history['val_acc'][0]) + '\n')
    output.close()
    return model