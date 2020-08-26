import os
import cv2
import tensorflow as tf

from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.tracker import Tracker
from .deep import model as Extractor


class DeepSortTracker(object):
    def __init__(
        self,
        model,
        classes,
        nn_budget=100,
        max_cosine_distance=0.5,
        nms_max_overlap=0.3,
        **tracker_kwargs
    ):
        """asdfasdfasdasdf"""

        self.model = model
        self.classes = classes

        # Load the feature extraction network from checkpoints
        parent_dir = os.path.dirname(__file__)
        self.extractor = Extractor([64, 128])
        self.extractor.load_weights(
            os.path.join(parent_dir, "checkpoints/extractor.tf")
        )

        # Configure the SORT tracker
        self.nms_max_overlap = nms_max_overlap
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, **tracker_kwargs)

    def _preprocess(self, img, size):
        """asdfasdfasdf"""
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = tf.image.resize(img_in, (size, size))
        img_in = img_in / 255

    def track(self, img, preprocess=None, size=416, **kwargs):
        """asdfasdfsd"""

        if preprocess:
            img_in = preprocess(img, **kwargs)

        else:
            img_in = self._preprocess(img, size)

        # Run the inference
        boxes, scores, classes, nums = self.model.predict(img_in)
