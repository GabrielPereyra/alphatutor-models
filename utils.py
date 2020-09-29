import os
import math
import chess
import numpy as np
import pandas as pd
import tensorflow as tf


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
    x = np.moveaxis(x, 1, 3)
    return x


class Data(tf.keras.utils.Sequence):

    def __init__(self, df, task, batch_size):
        self.df = df
        self.inputs = task['inputs']
        self.outputs = task['outputs']
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, i):
        i = np.random.randint(0, len(self.df), size=self.batch_size)
        # start = i * self.batch_size
        # end = i + self.batch_size
        # batch_df = self.df.iloc[start:end]
        batch_df = self.df.iloc[i]

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


def split(df, task, batch_size, train_size=0.8):
    train_size = int(len(df) * train_size)
    return Data(df[:train_size], task, batch_size), Data(df[train_size:], task, batch_size)
