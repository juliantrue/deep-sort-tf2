import os
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


from model import Model, Top
from dataset import load_train_dataset, load_test_dataset
from hessian_penalty import hessian_penalty

flags.DEFINE_string(
    "weights", "./deepsort/checkpoints/extractor_10.tf", "Path to weights"
)
flags.DEFINE_string("model_dir", "./deepsort/models", "Path to saved models")
flags.DEFINE_string(
    "train_dataset", "data/train", "Path to training dataset",
)
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("epochs", 2, "Number of epochs")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")


def transfer_weights(source, target):

    for sl in source.layers:
        for tl in target.layers:
            if sl.name == tl.name:
                print("assigning {} to {}".format(sl.name, tl.name))
                tl.set_weights(sl.get_weights())

    return target


def main(argv):
    logging.info("Loading training dataset...")
    train_dataset = load_train_dataset(FLAGS.train_dataset, FLAGS.batch_size)
    logging.info("Done!")

    # model = tf.keras.models.load_model("{}/extractor".format(FLAGS.model_dir))
    model = Model((FLAGS.img_height, FLAGS.img_width), bypass_top=True)
    model.load_weights(FLAGS.weights).expect_partial()
    top = Top(model.get_layer("cnn_output").output.shape)
    top = transfer_weights(model, top)

    # Iterate over epochs.
    optimizer = Adam(FLAGS.learning_rate)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    avg_loss = tf.keras.metrics.Mean()
    avg_hp_loss = tf.keras.metrics.Mean()
    for epoch in range(FLAGS.epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # Inference
                cnn_code = model(x_batch_train)
                inference = top(cnn_code, training=True)

                # Calculate loss
                sce_loss = loss_fn(tf.cast(y_batch_train, tf.float32), inference)
                hp_loss = hessian_penalty(top, cnn_code, training=True)
                loss = sce_loss + hp_loss

            # Apply gradients
            grads = tape.gradient(loss, top.trainable_weights)
            optimizer.apply_gradients(zip(grads, top.trainable_weights))
            avg_loss.update_state(sce_loss)
            avg_hp_loss.update_state(hp_loss)

            if step % 10 == 0:
                print(
                    "step {}: sce_loss = {:.4f}, hp_loss = {:.4f}".format(
                        step, avg_loss.result(), avg_hp_loss.result()
                    )
                )


if __name__ == "__main__":
    app.run(main)
