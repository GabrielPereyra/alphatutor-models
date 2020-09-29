import tensorflow as tf


mistake_from_elo = {
    'inputs': ['elo'],
    'outputs': {'mistake': tf.keras.losses.BinaryCrossentropy(from_logits=True)}
}


mistake_from_fen = {
    'inputs': ['fen'],
    'outputs': {'mistake': tf.keras.losses.BinaryCrossentropy(from_logits=True)}
}

mistake_from_fen_and_elo = {
    'inputs': ['fen', 'elo'],
    'outputs': {'mistake': tf.keras.losses.BinaryCrossentropy(from_logits=True)}
}

value_from_fen = {
    'inputs': ['fen'],
    'outputs': {'prev_score': 'mse'}
}