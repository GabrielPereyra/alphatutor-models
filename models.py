import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

# TODO: use this to check device placement on colab.
# tf.debugging.set_log_device_placement(True)
if tf.config.experimental.list_physical_devices('GPU'):
    tf.keras.backend.set_image_data_format('channels_first')


def Conv():
    return Conv2D(32, 3, activation='relu', padding='same')


def linear():
    inputs = x = Input(shape=(1,), name='elo')
    x = Dense(1, name='mistake')(x)
    return Model(inputs=inputs, outputs=x)


def conv(x):
    x = Conv()(x)
    x = MaxPool2D(2)(x)
    x = Conv()(x)
    x = MaxPool2D(2)(x)
    x = Conv()(x)
    x = GlobalMaxPool2D()(x)
    return x


# TODO: more general way to combine this with mistake conv?
def value_conv():
    inputs = Input(shape=(8, 8, 13), name='fen')
    x = conv(inputs)
    outputs = Dense(1, name='prev_score')(x)
    return Model(inputs=inputs, outputs=outputs)


def mistake_baseline():
    inputs = Input(shape=(1,), name='elo')
    outputs = Dense(1, name='mistake')(inputs)
    return Model(inputs=inputs, outputs=outputs)


def small_mistake_conv():
    inputs = Input(shape=(8, 8, 13), name='fen')
    x = conv(inputs)
    outputs = Dense(1, name='mistake')(x)
    return Model(inputs=inputs, outputs=outputs)


def small_mistake_elo_conv():
    fens = Input(shape=(8, 8, 13), name='fen')
    elos = Input(shape=(1,), name='elo')
    inputs = [fens, elos]
    x = conv(fens)
    x = Concatenate()([x, elos])
    outputs = Dense(1, name='mistake')(x)
    return Model(inputs=inputs, outputs=outputs)


def mistake_with_pretrained_value_conv():
    checkpoint_model = tf.keras.models.load_model('checkpoints/value-conv')
    layers = checkpoint_model.layers[1:-1]

    fens = x = Input(shape=(8, 8, 13), name='fen')
    for layer in layers:
        layer.trainable = False
        x = layer(x)

    outputs = Dense(1, name='mistake')(x)
    return Model(inputs=fens, outputs=outputs)


def small_elo_conv():
    inputs = Input(shape=(8, 8, 13), name='fen')
    x = conv(inputs)
    outputs = Dense(30, name='elo_bin')(x)
    return Model(inputs=inputs, outputs=outputs)


def small_elo_score_loss_conv():
    fens = Input(shape=(8, 8, 13), name='fen')
    score_loss = Input(shape=(1,), name='score_loss')
    inputs = [fens, score_loss]
    x = conv(fens)
    x = Concatenate()([x, score_loss])
    outputs = Dense(30, name='elo_bin')(x)
    return Model(inputs=inputs, outputs=outputs)


mistake_from_fen = {
    # TODO: baseline that takes a constant value and predicts mistake.
    'small': small_mistake_conv(),
}

mistake_from_fen_and_elo = {
    'baseline': mistake_baseline(),
    'small': small_mistake_elo_conv(),
}

elo_bin_from_fen = {
    'small': small_elo_conv(),
}

elo_bin_from_fen_and_score_loss = {
    'small': small_elo_score_loss_conv(),
}