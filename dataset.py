import tensorflow as tf
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def make_date(traindata_filepath, testdata_filepath, valdata_filepath):
    train_ds = tf.keras.utils.image_dataset_from_directory(traindata_filepath, image_size=(299, 299), batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(valdata_filepath, image_size=(299, 299), batch_size=32)
    test_ds = tf.keras.utils.image_dataset_from_directory(testdata_filepath, image_size=(299, 299), batch_size=32)
    class_names = train_ds.class_names

    # autotune = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    # val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    # normalization_layer = layers.Rescaling(1. / 255)
    #
    # def norm(x, y):
    #     return normalization_layer(x), y

    # train_ds = train_ds.map(norm)
    # val_ds = val_ds.map(norm)
    # test_ds = test_ds.map(norm)
    num_class = len(class_names)
    return train_ds, val_ds, test_ds, num_class


def parse_input(x, y):
    return x, tf.one_hot(y, 10)


def make_date(traindata_filepath, testdata_filepath,inpuy_shape_h, input_shape_w, channel):

    x_train = pd.read_csv(traindata_filepath)
    y_train = x_train['label']
    x_train = x_train.drop(columns='label')
    x_train = np.array(x_train)
    x_train = np.dstack([x_train] * channel)
    x_train = np.reshape(x_train, (-1, 28, 28, channel))

    x_train = np.asarray([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.array_to_img(im, scale=False).resize((inpuy_shape_h, input_shape_w))) for im in x_train])
    # print((np.shape(x_train)))
    x_train = x_train / 255.0
    x_train = x_train.astype('float32')
    classes = np.unique(y_train)
    num_classes = len(classes)
    # print(num_classes)
    y_train = to_categorical(y_train)
    x_test = pd.read_csv(testdata_filepath)
    y_test = x_test['label']
    x_test = x_test.drop(columns='label')
    x_test = np.array(x_test)
    x_test = np.dstack([x_test] * channel)
    x_test = np.reshape(x_test, (-1, 28, 28, channel))
    x_test = np.asarray([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.array_to_img(im, scale=False).resize((inpuy_shape_h, input_shape_w))) for im in x_test])
    # print((np.shape(x_test)))
    x_test = x_test / 255.0
    x_test = x_test.astype('float32')
    y_test = to_categorical(y_test)

    # print(type(x_train))
    # print(np.shape(x_train))
    #
    # print(type(y_train))
    # print(np.shape(y_train))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=13)
    # x_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # x_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # x_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # x_train = x_train.shuffle(1000).batch(64)
    # x_test = x_test.shuffle(1000).batch(64)
    # x_val = x_val.shuffle(1000).batch(64)

    # print one image
    # plt.imshow(x_train[10])
    # plt.show()

    return x_train, y_train, x_val, y_val, x_test, y_test, num_classes
