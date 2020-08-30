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


flags.DEFINE_string(
    "train_dataset", "data/train", "Path to training dataset",
)
flags.DEFINE_string(
    "test_dataset", "data/test", "Path to testing dataset",
)
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer("img_width", 64, "image width")
flags.DEFINE_integer("img_height", 128, "image height")
flags.DEFINE_boolean(
    "eager", False, "Execute training with gradient tape (True) or graph mode (False)",
)


def transform_and_augment_images(x_train, size):
    re_size = [size[0] + 5, size[1] + 5]
    x_train = tf.image.resize(x_train, re_size, method="bicubic")
    x_train = x_train / 255
    x_train = tf.image.random_flip_left_right(x_train)
    x_train = tf.image.random_jpeg_quality(x_train, 50, 95)
    x_train = tf.image.random_crop(x_train, size=[size[0], size[1], 3])
    return x_train


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, size, method="bicubic")
    x_train = x_train / 255
    return x_train


def parse_tfrecord(example_proto, feature_description):
    # Parse the input tf.Example proto using the dictionary above.
    example = tf.io.parse_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(example["image/encoded"])
    label = example["label"]
    return image, label


def main(argv):
    feature_description = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
    }

    logging.info("Loading training dataset...")
    train_filenames = [
        os.path.join(FLAGS.train_dataset, path)
        for path in os.listdir(FLAGS.train_dataset)
    ]

    test_filenames = [
        os.path.join(FLAGS.test_dataset, path)
        for path in os.listdir(FLAGS.test_dataset)
    ]
    filenames = train_filenames + test_filenames
    full_dataset = tf.data.TFRecordDataset(filenames=filenames)
    full_dataset = full_dataset.map(lambda x: parse_tfrecord(x, feature_description))
    full_dataset = full_dataset.shuffle(buffer_size=8192)

    # Break the dataset into training, testing
    # 80/20 split
    def is_test(x, y):
        return x % 5 == 0

    def is_train(x, y):
        return not is_test(x, y)

    recover = lambda x, y: y

    test_dataset = full_dataset.enumerate().filter(is_test).map(recover)
    train_dataset = full_dataset.enumerate().filter(is_train).map(recover)

    train_dataset = train_dataset.map(
        lambda x, y: (
            transform_and_augment_images(x, [FLAGS.img_height, FLAGS.img_width]),
            y,
        )
    )
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    # logging.info("Done!")

    # logging.info("Loading testing dataset...")
    # test_dataset = tf.data.TFRecordDataset(
    #    filenames=[
    #        os.path.join(FLAGS.test_dataset, path)
    #        for path in os.listdir(FLAGS.test_dataset)
    #    ]
    # )
    # test_dataset = test_dataset.shuffle(buffer_size=4096)
    # test_dataset = test_dataset.map(lambda x: parse_tfrecord(x, feature_description))
    test_dataset = test_dataset.map(
        lambda x, y: (transform_images(x, [FLAGS.img_height, FLAGS.img_width]), y)
    )
    test_dataset = test_dataset.batch(FLAGS.batch_size)
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
