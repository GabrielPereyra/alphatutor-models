import shutil
import click
import utils
import tasks
import models
import datetime
import tensorflow as tf


def get_callbacks(name):
    shutil.rmtree(name, ignore_errors=True)
    return tf.keras.callbacks.TensorBoard(
        name,
        profile_batch=0,
        update_freq=100,
    )


@click.command()
@click.argument('task')
@click.argument('size')
@click.option('--data', default='data/csvs')
@click.option('--shards', default=1)
@click.option('--checkpoint', is_flag=True)
def train(task, size, data, shards, checkpoint):
    name = '{}/{}-{}/'.format(task, size, shards)
    model = getattr(models, task)[size]
    task = getattr(tasks, task)

    df = utils.get_df(data, shards)
    df = utils.normalize_df(df)
    df = df.sample(frac=1)
    dataset = utils.Data(df, task)
    callbacks = get_callbacks('logs/{}'.format(name))

    model.compile(optimizer='adam', loss=task['outputs'], metrics=task.get('metrics'))
    model.summary()

    model.fit(
        dataset,
        callbacks=callbacks,
        workers=2,
        max_queue_size=10,
        use_multiprocessing=True,
    )

    if checkpoint:
        model.save('checkpoints/{}'.format(name))


if __name__ == '__main__':
    train()