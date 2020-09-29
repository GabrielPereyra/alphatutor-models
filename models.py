from tensorflow.keras.layers import *
from tensorflow.keras import Model


def linear():
    inputs = x = Input(shape=(1,), name='elo')
    x = Dense(1, name='mistake')(x)
    return Model(inputs=inputs, outputs=x)


def elo_conv():
    fens = Input(shape=(8, 8, 13), name='fen')
    elos = Input(shape=(1,), name='elo')
    inputs = [fens, elos]
    x = Conv2D(32, 3, padding='same')(fens)
    x = MaxPool2D(2)(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = GlobalMaxPool2D()(x)
    x = Concatenate()([x, elos])
    x = Dense(1, name='mistake')(x)
    return Model(inputs=inputs, outputs=x)


def conv(x):
    x = Conv2D(32, 3, padding='same')(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = GlobalMaxPool2D()(x)
    return x


def value_conv():
    inputs = Input(shape=(8, 8, 13), name='fen')
    x = conv(inputs)
    outputs = Dense(1, name='prev_score')(x)
    return Model(inputs=inputs, outputs=outputs)


def mistake_conv():
    inputs = Input(shape=(8, 8, 13), name='fen')
    x = conv(inputs)
    outputs = Dense(1, name='mistake')(x)
    return Model(inputs=inputs, outputs=outputs)
