import tensorflow as tf
import math
import matplotlib.pyplot as plt


def plot_graphslr(history, metric):
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plot_graphs(history_model, 'accuracy')
    # plt.subplot(1, 3, 2)
    # plot_graphslr(history_model, 'lr')
    # plt.subplot(1, 3, 3)
    # plot_graphs(history_model, 'loss')
    # plt.show()


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def model_callbacks():
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='D:\PyCharm\DogBreed\checkpoint\checkpoint',
        save_weights_only=True,
        save_best_only=True,
        monitor='val_acc',
        mode='max', )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')

    initial_learning_rate = 0.001

    def lr_step_decay(epoch, lr):
        # learning rate schedule whit step decay
        drop_rate = 0.5
        epoch_drop = 5
        return initial_learning_rate * math.pow(drop_rate, math.floor(epoch / epoch_drop))

    def lr_time_based_decay(epoch, lr):
        # learning rate schedule whit time based decay
        decay = initial_learning_rate / epoch
        return lr * 1 / (1 + decay * epoch)

    def lr_exp_decay(epoch, lr):
        # learning rate schedule whit Exponential decay
        k = 0.1
        return initial_learning_rate * math.exp(-k * epoch)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1)
    return model_checkpoint_callback, stop_early, lr_schedule
