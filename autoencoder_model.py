import tensorflow as tf


def autoencoding_model_1(input_shape_h, input_shape_w, channel):
    input_layer = tf.keras.layers.Input(shape=(input_shape_h, input_shape_w, channel))
    x = tf.keras.layers.Conv2D()(input_layer)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    encoded = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D()(encoded)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D()(x)
    decoded = tf.keras.layers.MaxPooling2D()(x)

    autoencoding_models = tf.keras.Model(inputs=input_layer, outputs=decoded)
    autoencoding_models.summary(show_trainable='True', expand_nested='True')
    autoencoding_models.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=["acc"])
    print('autoencoder_model')
    return autoencoding_models


def autoencoding_model(input_shape_h, input_shape_w, channel):

    input_layer = tf.keras.Input(shape=(input_shape_h, input_shape_w, channel))

    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder_model = tf.keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder_model.summary(show_trainable='True', expand_nested='True')
    autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=["acc"])
    print('autoencoder_model')

    return autoencoder_model
