import utils
import numpy as np
import tensorflow as tf
import plotly.express as px



# TODO: compare small-1, small-5, small-10 to see if improving model improves performance.

df = utils.get_df('data/csvs', 1)[:10000]
df['count'] = 1
model = tf.keras.models.load_model('checkpoints/elo_bin_from_fen/small-10')


def elo_bin(elos):
    return int(elos.mean() / 100)


def estimate_elo(fens):
    x = utils.fens_to_array(fens)
    logits = model.predict(x) * 2
    probs = tf.keras.layers.Softmax()(logits)
    prob = tf.math.reduce_mean(probs, axis=0)
    elo_bin = tf.argmax(prob, axis=-1).numpy()
    return elo_bin


df = df.groupby('username').agg({'fen': estimate_elo, 'elo': elo_bin, 'count': 'count'})

df['log_count'] = np.log10(df['count']).astype(int)
df['correct'] = df['fen'] == df['elo']

import pdb; pdb.set_trace()

fig = px.scatter(df, x="count", y="correct", trendline="ols")
fig.show()


# print(tf.keras.metrics.SparseCategoricalAccuracy()(y_true, y_pred).numpy())
# print(tf.keras.metrics.SparseCategoricalCrossentropy()(y_true, y_pred).numpy())