from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy


mistake_from_elo = {
    'inputs': ['elo'],
    'outputs': {'mistake': BinaryCrossentropy(from_logits=True)}
}


mistake_from_fen = {
    'inputs': ['fen'],
    'outputs': {'mistake': BinaryCrossentropy(from_logits=True)},
    'metrics': 'accuracy',
}

mistake_from_fen_and_elo = {
    'inputs': ['fen', 'elo'],
    'outputs': {'mistake': BinaryCrossentropy(from_logits=True)},
    'metrics': 'accuracy',
}

value_from_fen = {
    'inputs': ['fen'],
    'outputs': {'prev_score': 'mse'}
}

elo_bin_from_fen = {
    'inputs': ['fen'],
    'outputs': {'elo_bin': SparseCategoricalCrossentropy(from_logits=True)},
    'metrics': 'accuracy',
}

elo_bin_from_fen_and_score_loss = {
    'inputs': ['fen', 'score_loss'],
    'outputs': {'elo_bin': SparseCategoricalCrossentropy(from_logits=True)},
    'metrics': 'accuracy',
}