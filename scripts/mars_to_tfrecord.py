import sys, os
import random
from absl import app, flags
from absl.flags import FLAGS

import tensorflow as tf
from tqdm import tqdm

flags.DEFINE_string(
    "raw_train_folder", "../data/mars/bbox_train", "Path to MARS bbox_train."
)
flags.DEFINE_string(
    "raw_test_folder", "../data/mars/bbox_test", "Path to MARS bbox_test."
)
flags.DEFINE_string(
    "target_train_folder",
    "../data/train",
    "Path to folder to store train tfrecords in.",
)
flags.DEFINE_string(
    "target_val_folder",
    "../data/val",
    "Path to folder to store validation tfrecords in.",
)
flags.DEFINE_string(
    "target_test_folder", "../data/test", "Path to folder to store test tfrecords in.",
)

flags.DEFINE_list(
    "train_val_test",
    [0.7, 0.15, 0.15],
    "List of percentages of the dataset to be attributed to training, validation, and testing respectively.",
)


def read_train_test_directory_to_str(directory):
    """Read bbox_train/bbox_test directory.
    
    Args:
        directory (str): Path to bbox_train/bbox_test directory.

    Returns (List[str], List[int], List[int], List[int]):
        Returns a tuple with the following entries:
        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.
    """

    def to_label(x):
        return int(x) if x.isdigit() else -1

    dirnames = os.listdir(directory)
    image_filenames, ids, camera_indices, tracklet_indices = [], [], [], []
    for i, dirname in tqdm(enumerate(dirnames)):
        if not dirname.isdigit():
            continue

        filenames = os.listdir(os.path.join(directory, dirname))
        filenames = [f for f in filenames if os.path.splitext(f)[1] == ".jpg"]
        image_filenames += [os.path.join(directory, dirname, f) for f in filenames]
        ids += [to_label(dirname) for _ in filenames]
        camera_indices += [int(f[5]) for f in filenames]
        tracklet_indices += [int(f[7:11]) for f in filenames]

    return image_filenames, ids, camera_indices, tracklet_indices


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte.
    https://www.tensorflow.org/tutorials/load_data/tfrecord#writing_a_tfrecord_file_2
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint.
    https://www.tensorflow.org/tutorials/load_data/tfrecord#writing_a_tfrecord_file_2
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tf_example(image_filename, label):
    """Converts image and label to a tf Example protobuf  
    https://www.tensorflow.org/tutorials/load_data/tfrecord#writing_a_tfrecord_file_2

    Args:
        image_string (str): String representation of jpeg image
        label (list): tbd
    """
    image_string = open(image_filename, "rb").read()
    image = _bytes_feature(image_string)
    label = _int64_feature(label)
    feature = {"image/encoded": image, "label": label}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_dataset_to_tf_records(record_filename, data, imgs_per_shard=1000):
    """Convert a list of tuples (image_filename, label) into a tfrecord dataset"""
    wo_extension = record_filename.split(".")[:-1]
    wo_extension[-1] += "_{}"
    wo_extension = ".".join(wo_extension)
    record_filename = wo_extension + ".tfrecords"
    shard = 0
    writer = tf.io.TFRecordWriter(record_filename.format(shard))
    for i, img_filename_label in tqdm(enumerate(data)):
        img_filename, label = img_filename_label
        if i % imgs_per_shard == 0:
            writer.close()
            writer = tf.io.TFRecordWriter(record_filename.format(i))

        tf_example = to_tf_example(img_filename, label)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(argv):

    # Check if there are already generated tfrecords in the target directory
    if (
        len(os.listdir(FLAGS.target_train_folder)) > 0
        or len(os.listdir(FLAGS.target_test_folder)) > 0
    ):
        print(
            "Files already exist in the target training or testing folder: {}".format(
                FLAGS.target_train_folder
            )
        )
        resp = input("Do you want to continue? (y/n) ").lower()

        if resp == "y":
            print("TODO: write this branch to remove existing files.")

        else:
            print("Exiting.")
            sys.exit(0)

    # Get the two datasets two encode
    # Returns image_filenames, ids, camera_indices, tracklet_indices
    # We only need the image_filenames and the ids
    print("Reading training dataset.")
    train_image_filenames, train_ids, _, _ = read_train_test_directory_to_str(
        FLAGS.raw_train_folder
    )

    print("Reading testing dataset.")
    test_image_filenames, test_ids, _, _ = read_train_test_directory_to_str(
        FLAGS.raw_test_folder
    )

    # Suffle in place and apply train, val, test split
    print("Shuffling data")
    data = list(zip(train_image_filenames, train_ids)) + list(
        zip(test_image_filenames, test_ids)
    )
    random.shuffle(data)
    num_points = len(data)
    idx = range(num_points)
    print("Done!")

    print("Splitting dataset according to {}/{}/{}".format(*FLAGS.train_val_test))
    max_train_idx = int(num_points * FLAGS.train_val_test[0])
    max_val_idx = int(num_points * FLAGS.train_val_test[1])
    min_test_idx = int(num_points * FLAGS.train_val_test[2])
    train = data[:max_train_idx]
    val = data[max_train_idx:max_val_idx]
    test = data[min_test_idx:]

    print("Training indices [0:{})".format(max_train_idx))
    print("Validation indices [{}:{})".format(max_train_idx, max_val_idx))
    print("Testing indices [{}:{}]".format(min_test_idx, num_points))
    print("Done!")

    # Put one more check in place to make sure that the existing data is removed
    if (
        len(os.listdir(FLAGS.target_train_folder)) > 0
        or len(os.listdir(FLAGS.target_test_folder)) > 0
    ):
        print("Training or testing folder is not empty.")
        print("Please empty it before continuing.")
        print("Exiting")
        sys.exit(0)

    # Currently this writes only one tf records file.
    # If for some reason this becomes a problem, a "num_parts" field will be defined
    # to break the file into many parts
    #
    # Runtime eta ~
    print("Writing training dataset to tf record files...")
    train_dataset_path = os.path.join(FLAGS.target_train_folder, "train.tfrecords")
    write_dataset_to_tf_records(train_dataset_path, train)

    # Runtime eta ~
    print("Writing validation dataset to tf record files...")
    val_dataset_path = os.path.join(FLAGS.target_val_folder, "val.tfrecords")
    write_dataset_to_tf_records(val_dataset_path, val)

    # Runtime eta ~
    print("Writing testing dataset to tf record files...")
    test_dataset_path = os.path.join(FLAGS.target_test_folder, "test.tfrecords")
    write_dataset_to_tf_records(test_dataset_path, test)

    print("Dataset generation complete.")
    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
