import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import dataset
import autoencoder_model
import matplotlib.pyplot as plt
import utility

if __name__ == '__main__':
    input_shape_given_h = 28
    input_shape_given_w = 28
    channel_given = 1
    batch_size_given = 128
    epoch_given = 20

    x_train, y_train, x_val, y_val, x_test, y_test, num_class = dataset.make_date(
        "D:\PyCharm\\fashion_mnist\\fashion-mnist_train.csv", "D:\PyCharm\\fashion_mnist\\fashion-mnist_test.csv",
        input_shape_given_h, input_shape_given_w, channel_given)

    model = autoencoder_model.autoencoding_model(input_shape_given_h, input_shape_given_w, channel_given)
    model_checkpoint_callback, stop_early, lr_schedule = utility.model_callbacks()
    history = model.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=epoch_given,
                        batch_size=batch_size_given,
                        callbacks=[stop_early, model_checkpoint_callback, lr_schedule])

    # load saved state of model to start new traing with best model that have been saved in chechpoint from previous
    # model fitting
    # model.load_weights('D:\PyCharm\DogBreed\checkpoint\checkpoint')
    # new_history = model.fit(train_data, epochs=5, validation_data=val_data,
    #                         callbacks=[stop_early, model_checkpoint_callback], )

    resulats = model.evaluate(x_test, x_test, verbose=1, batch_size=batch_size_given)
    model.save('D:\PyCharm\DogBreed\saved_model')
    prediction = model.predict(x_test)
    # print(prediction.tolist())
    # print(np.argmax(prediction[:32], axis=1))
    # print(list(x_test.take(1).as_numpy_iterator())[0][1])
    # print('result:::::', resulats)
    # print(type(resulats))

    # plt.plot(history.history["acc"])
    # plt.plot(history.history['val_acc'])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title("model accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Epoch")
    # plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    # plt.show()

    # print(history.history.keys())
    # print(type(history))
    print('loss:', max(history.history['loss']))
    print('acc:', max(history.history['acc']))
    print('val_loss:', max(history.history['val_loss']))
    print('val_acc:', max(history.history['val_acc']))
    print('test_loss:', resulats[0])
    print('test_acc:', resulats[1])
    print('input_shape:', input_shape_given_h, input_shape_given_w, '    channel:', channel_given,
          '    batch_size:', batch_size_given, '    epoch:', epoch_given, )
