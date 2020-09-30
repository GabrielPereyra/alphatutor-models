import numpy as np
import pandas as pd
from datasketch import MinHashLSHForest, MinHash

# TODO: compare this with average score loss
# TODO: neither of these work well with puzzles...

df = pd.read_csv('../chess-opening/csvs/lichess_db_standard_rated_2020-08_600+0.csv', nrows=1000000)


def create_min_hash(fens):
    min_hash = MinHash(num_perm=128)
    for fen in fens:
        min_hash.update(fen.encode('utf8'))
    return min_hash


user_df = df.groupby('username').agg({'fen': set, 'elo': 'mean'})
user_df['min_hash'] = user_df['fen'].apply(create_min_hash)

forest = MinHashLSHForest(num_perm=128)
for row in user_df.itertuples():
    forest.add(row.Index, row.min_hash)


forest.index()


for i in range(10):
    result = forest.query(user_df['min_hash'][i], 10)

    elos = []
    for username in result:
        elos.append(user_df.loc[username]['elo'])
    print(user_df['elo'][i], np.mean(elos))
