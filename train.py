import click
import utils
import tasks
import models
import tensorflow as tf


@click.command()
@click.argument('task')
@click.argument('model')
@click.option('--shards', default=1)
@click.option('--checkpoint')
@click.option('--tensorboard')
def train(task, model, shards, checkpoint, tensorboard):
    task = getattr(tasks, task)
    model = getattr(models, model)()

    df = utils.get_df(shards)
    train_data, valid_data = utils.split(df, task, 1024)
    model.compile(optimizer='adam', loss=task['outputs'])
    model.summary()
    model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=tf.keras.callbacks.TensorBoard(tensorboard) if tensorboard else None)

    if checkpoint:
        model.save('checkpoints/{}'.format(checkpoint))


# TODO: how to train a model that requires a checkpoint?
# TODO: how to store tensorboard?
# TODO: how to set checkpoint path (need this for colab).


if __name__ == '__main__':
    train()