import click
import chess
import chess.pgn
import chess.engine
import pandas as pd
import itertools
time_control = '600+0'
shard_size = 100000


def game_to_rows(game):
    rows = []
    prev_score = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
    for node in game.mainline():
        score = node.eval()
        if score is None:
            break

        turn = node.parent.board().turn
        username = game.headers["White"] if turn else game.headers["Black"]
        elo = game.headers["WhiteElo"] if turn else game.headers["BlackElo"]

        row = {
            "elo": elo,
            "username": username,
            "opening_string": game.headers["Opening"],
            "eco": game.headers["ECO"],
            "fen": node.parent.board().fen(),
            "move": node.uci(),
            "clock": node.clock(),
            "score": score.pov(turn).score(),
            "mate": score.pov(turn).mate(),
            "prev_score": prev_score.pov(turn).score(),
            "prev_mate": prev_score.pov(turn).mate(),
            "fullmove_number": node.parent.board().fullmove_number,
            "turn": turn,
        }

        rows.append(row)
        prev_score = score

    return rows


def get_offsets(pgn, limit):
    offsets = []

    for i in itertools.count():
        if i % 10000 == 0:
            print('{:>10} {:>10}'.format(i, len(offsets)))

        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)

        if headers is None:
             break
        if headers['TimeControl'] != time_control:
            continue
        if headers["WhiteElo"] == "?":
            continue
        if headers["BlackElo"] == "?":
            continue
        if limit and len(offsets) >= limit:
            break

        offsets.append(offset)
    return offsets


def offsets_to_df(offsets, pgn):
    print('Parsing {} games.'.format(len(offsets)))

    rows = []
    with click.progressbar(offsets, label="Parsing pgn") as offsets:
        for offset in offsets:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            rows.extend(game_to_rows(game))

    return pd.DataFrame(rows)


@click.command()
@click.argument('path')
@click.option('--limit', type=int)
def parse(path, limit):
    pgn = open(path)
    offsets = get_offsets(pgn, limit)

    for i in itertools.count(0):
        start = i * shard_size
        end = start + shard_size
        df = offsets_to_df(offsets[start:end], pgn)
        csv_path = path.replace('pgn', 'csv').replace('.', '/{}.'.format(i))

        df.to_csv(csv_path, index=False)

        if end >= len(offsets):
            break


if __name__ == '__main__':
    parse()