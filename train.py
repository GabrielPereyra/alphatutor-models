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
    train_data, valid_data = utils.split(df, task, 1024)
    model.compile(optimizer='adam', loss=task['outputs'])
    model.summary()
    model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=tf.keras.callbacks.TensorBoard(tensorboard) if tensorboard else None)

    if checkpoint:
        model.save('checkpoints/{}'.format(checkpoint))


# TODO: how to train a model that requires a checkpoint?


if __name__ == '__main__':
    train()