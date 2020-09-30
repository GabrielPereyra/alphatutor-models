import shutil
import click
import utils
import tasks
import models
import tensorflow as tf


@click.command()
@click.argument('task')
@click.argument('model')
@click.option('--data', default='data/csvs')
@click.option('--shards', default=1)
@click.option('--checkpoint')
@click.option('--tensorboard')
def train(task, model, data, shards, checkpoint, tensorboard):
    task = getattr(tasks, task)
    model = getattr(models, model)()

    df = utils.get_df(data, shards)
    df = utils.normalize_df(df)
    df = df.sample(frac=1)
    dataset = utils.Data(df, task)

    callbacks = []
    if tensorboard:
        shutil.rmtree(tensorboard, ignore_errors=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(
            tensorboard,
            profile_batch=2,
            update_freq=100,
        ))

    model.compile(optimizer='adam', loss=task['outputs'], metrics=task.get('metrics'))
    model.summary()

    model.fit(
        dataset,
        callbacks=callbacks,
        workers=4,
        max_queue_size=1000,
        use_multiprocessing=True,
    )

    if checkpoint:
        model.save('checkpoints/{}'.format(checkpoint))


# TODO: how to train a model that requires a checkpoint? Load from checkpoint to start.


if __name__ == '__main__':
    train()