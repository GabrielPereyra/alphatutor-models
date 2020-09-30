import os
import math
import chess
import numpy as np
import pandas as pd
import tensorflow as tf
GPU = bool(tf.config.experimental.list_physical_devices('GPU'))



def get_df(data, shards):
    path = '{}/lichess_db_standard_rated_2020-08/'.format(data)
    dfs = []
    for i, filename in enumerate(os.listdir(path)):
        if i >= shards:
            break
        df = pd.read_csv(path + filename)
        dfs.append(df)
    return pd.concat(dfs)


def normalize_df(df):
    df = df[~df['prev_score'].isna()]
    df['prev_score'] = (df['prev_score'] - df['prev_score'].mean()) / df['prev_score'].std()
    df['score_loss'] = df['prev_score'] - df['score']
    df = df[~df['score_loss'].isna()]
    df['mistake'] = df['score_loss'] > 50
    df['elo'] = (df['elo'] - df['elo'].mean()) / df['elo'].std()
    return df


def fen_to_bitboard(fen):
    # fen = fen.numpy().decode("utf-8")
    board = chess.Board(fen)
    bs = []
    for color in [board.turn, not board.turn]:
        for piece_type in chess.PIECE_TYPES:
            bs.append(board.pieces(piece_type, color).tolist())
    bs.append([board.turn] * 64)
    return bs


def fens_to_array(fens):
    x = []
    for fen in fens:
        x.append(fen_to_bitboard(fen))
    x = np.array(x)
    x = x.reshape(-1, 13, 8, 8)
    if not GPU:
        x = np.moveaxis(x, 1, 3)
    return x


class Data(tf.keras.utils.Sequence):

    def __init__(self, df, task, batch_size=1024):
        self.df = df
        self.inputs = task['inputs']
        self.outputs = task['outputs']
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, i):
        # i = np.random.randint(0, len(self.df), size=self.batch_size)
        # batch_df = self.df.iloc[i]

        start = i * self.batch_size
        end = start + self.batch_size
        batch_df = self.df.iloc[start:end]

        inputs = {}
        for input in self.inputs:
            if input == 'fen':
                inputs[input] = fens_to_array(batch_df[input])
            else:
                inputs[input] = batch_df[input].values

        outputs = {}
        for output in self.outputs:
            outputs[output] = batch_df[output].values

        return inputs, outputs


# def map_func(x, y):
#     x['fen'] = tf.py_function(fen_to_bitboard, [x['fen']], tf.float32)
#     return x, y
#
#
# def df_to_dataset(df, task):
#     input_dict = {}
#     for input in task['inputs']:
#         input_dict[input] = df[input]
#     output_dict = {}
#     for output in task['outputs']:
#         output_dict[output] = df[output]
#
#     dataset = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))
#     dataset = dataset.map(
#         map_func,
#         num_parallel_calls=4,
#         deterministic=False
#     )
#     dataset = dataset.batch(1024) # TODO: good size for shuffle buffer?
#     dataset = dataset.prefetch(100)  # TODO: set prefetch shapes.
#     return dataset


def split(df, task, batch_size, train_size=0.8):
    train_size = int(len(df) * train_size)
    return Data(df[:train_size], task, batch_size), Data(df[train_size:], task, batch_size)
