import os
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)

from model import Model
from dataset import load_train_dataset, load_test_dataset


flags.DEFINE_string(
    "train_dataset", "data/train", "Path to training dataset",
)
flags.DEFINE_string(
    "test_dataset", "data/test", "Path to testing dataset",
)
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_boolean(
    "eager", False, "Execute training with gradient tape (True) or graph mode (False)",
)


def main(argv):
    logging.info("Loading training dataset...")
    train_dataset = load_train_dataset(FLAGS.train_dataset, FLAGS.batch_size)
    logging.info("Done!")

    logging.info("Loading test dataset...")
    test_dataset = load_test_dataset(FLAGS.test_dataset, FLAGS.batch_size)
    logging.info("Done!")

    logging.info("Creating model and starting training.")
    model = Model((FLAGS.img_height, FLAGS.img_width), num_classes=1501, training=True)

    if FLAGS.eager:
        optimizer = Adam(FLAGS.learning_rate)
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        acc = SparseCategoricalAccuracy()
        avg_acc = tf.keras.metrics.Mean()
        avg_loss = tf.keras.metrics.Mean()
        avg_val_loss = tf.keras.metrics.Mean()

        # Iterate over epochs.
        import numpy as np

        for epoch in range(FLAGS.epochs):
            print("Start of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # Inference
                    inference = model(x_batch_train)

                    # Calculate loss
                    loss = loss_fn(tf.cast(y_batch_train, tf.float32), inference)

                # Apply gradients
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                avg_loss.update_state(loss)

                if step % 10 == 0:
                    print("step {}: loss = {:.4f}".format(step, avg_loss.result()))

            # Validation step.
            for step, (x_batch_val, y_batch_val) in enumerate(test_dataset.take(1000)):

                # Inference
                inference = model.predict(x_batch_val)

                # Calculate loss
                loss = loss_fn(tf.cast(y_batch_val, tf.float32), inference)

                # Calculate accuracy
                accuracy = acc(y_batch_val, inference)

                # Add to running average
                avg_val_loss.update_state(loss)
                avg_acc.update_state(accuracy)

                if step % 10 == 0:
                    print(
                        "step {}: loss = {:.4f}, acc = {:.4f}".format(
                            step, avg_val_loss.result(), avg_acc.result()
                        )
                    )
            avg_loss.reset_state()
            avg_val_loss.reset_state()
            avg_acc.reset_state()

    else:

        def scheduler(epoch):

            if epoch < 3:
                return FLAGS.learning_rate

            elif epoch < 7:
                return 1e-4

            else:
                return 1e-5

        model.compile(
            optimizer=Adam(FLAGS.learning_rate),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()],
        )

        callbacks = [
            LearningRateScheduler(scheduler, verbose=1),
            ModelCheckpoint("checkpoints/cml_{epoch}.tf", save_weights_only=True),
            TensorBoard(log_dir="logs", histogram_freq=1, update_freq=1000),
        ]

        history = model.fit(
            train_dataset,
            epochs=FLAGS.epochs,
            callbacks=callbacks,
            validation_data=test_dataset,
            verbose=1,
        )


if __name__ == "__main__":
    app.run(main)
