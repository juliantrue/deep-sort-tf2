import os
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)

from tensorboard.plugins.hparams import api as hp

from model import Model
from dataset import load_train_dataset, load_test_dataset


flags.DEFINE_string(
    "train_dataset", "data/train", "Path to training dataset",
)
flags.DEFINE_string(
    "test_dataset", "data/test", "Path to testing dataset",
)
flags.DEFINE_boolean(
    "hessian_penalty", False, "Train with the hessian_penalty on the last layer"
)
flags.DEFINE_string("logdir", "logs", "Path to logdir")
flags.DEFINE_string(
    "checkpoint_dir", "./deepsort/checkpoints", "Path to checkpoints directory"
)
flags.DEFINE_string("model_dir", "./deepsort/models", "Path to write trained model to.")
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_enum(
    "mode",
    "graph",
    ["eager", "graph", "hyperparameter"],
    (
        "Execute training with gradient tape training, graph mode training, or full "
        "hyperparameter grid search."
    ),
)


def main(argv):
    mem_limit = 8000
    logging.info("Setting Max Memory Usage to: {}GB".format(mem_limit / 1000))
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)],
    )

    logging.info("Loading training dataset...")
    train_dataset = load_train_dataset(FLAGS.train_dataset, FLAGS.batch_size)
    logging.info("Done!")

    logging.info("Loading test dataset...")
    test_dataset = load_test_dataset(FLAGS.test_dataset, FLAGS.batch_size)
    logging.info("Done!")

    logging.info("Creating model and starting training.")
    if FLAGS.mode == "eager":
        model = Model(
            (FLAGS.img_height, FLAGS.img_width), num_classes=1501, training=True
        )
        optimizer = Adam(FLAGS.learning_rate)
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        acc = SparseCategoricalAccuracy()
        avg_acc = tf.keras.metrics.Mean()
        avg_loss = tf.keras.metrics.Mean()
        avg_hp_loss = tf.keras.metrics.Mean()
        avg_val_loss = tf.keras.metrics.Mean()

        # Iterate over epochs.
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
                avg_hp_loss.update_state(loss)

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
                        "step {}: loss = {:.4f}, hp_loss={:.4f}, acc = {:.4f}".format(
                            step,
                            avg_val_loss.result(),
                            avg_hp_loss.result(),
                            avg_acc.result(),
                        )
                    )

            avg_hp_loss.reset_state()
            avg_loss.reset_state()
            avg_val_loss.reset_state()
            avg_acc.reset_state()

    elif FLAGS.mode == "graph":
        model = Model(
            (FLAGS.img_height, FLAGS.img_width), num_classes=1501, training=True
        )

        # def scheduler(epoch):

        #    if epoch < 3:
        #        return FLAGS.learning_rate

        #    elif epoch < 7:
        #        return 1e-4

        #    else:
        #        return 1e-5

        model.compile(
            optimizer=Adam(FLAGS.learning_rate),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()],
        )

        callbacks = [
            # LearningRateScheduler(scheduler, verbose=1),
            ReduceLROnPlateau(patience=2),
            ModelCheckpoint(
                "{}/".format(FLAGS.checkpoint_dir) + "extractor_{epoch}.tf",
                save_weights_only=True,
            ),
            TensorBoard(log_dir=FLAGS.logdir, histogram_freq=1, update_freq=1000),
        ]

        history = model.fit(
            train_dataset,
            epochs=FLAGS.epochs,
            callbacks=callbacks,
            validation_data=test_dataset,
            verbose=1,
        )

        model.save("{}/extractor".format(FLAGS.model_dir))

    elif FLAGS.mode == "hyperparameter":

        # Declare the ranges of the sweep
        HP_LR_TYPES = hp.HParam("lr_type", hp.Discrete(["static", "dynamic"]))
        HP_LRS = hp.HParam("learning_rate", hp.Discrete([1e-2, 1e-3]))
        HP_EPOCHS = hp.HParam("epochs", hp.Discrete([5, 10, 15]))

        def run(hparams):
            model = Model(
                (FLAGS.img_height, FLAGS.img_width), num_classes=1501, training=True
            )
            model.compile(
                optimizer=Adam(hparams[HP_LRS]),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=[SparseCategoricalAccuracy()],
            )

            if hparams[HP_LR_TYPES] == "static":
                callbacks = [
                    TensorBoard(log_dir=FLAGS.logdir),
                    hp.KerasCallback(FLAGS.logdir, hparams),  # log hparams
                ]

            else:
                callbacks = [
                    ReduceLROnPlateau(patience=3),
                    TensorBoard(log_dir=FLAGS.logdir),
                    hp.KerasCallback(FLAGS.logdir, hparams),  # log hparams
                ]

            history = model.fit(
                train_dataset,
                epochs=hparams[HP_EPOCHS],
                callbacks=callbacks,
                validation_data=test_dataset,
            )

        # Perform the grid search
        session_num = 0
        for lr_type in HP_LR_TYPES.domain.values:
            for lr in HP_LRS.domain.values:
                for epochs in HP_EPOCHS.domain.values:
                    hparams = {
                        HP_LR_TYPES: lr_type,
                        HP_LRS: lr,
                        HP_EPOCHS: epochs,
                    }

                    run_name = "run-%d" % session_num
                    print("--- Starting trial: %s" % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run(hparams)
                    session_num += 1


if __name__ == "__main__":
    app.run(main)
