import os
import cv2
import numpy as np
import tensorflow as tf

from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.tracker import Tracker
from .deep import Model as Extractor


class DeepSortTracker(object):
    def __init__(
        self,
        model=None,
        nn_budget=100,
        max_cosine_distance=0.5,
        nms_max_overlap=0.3,
        min_confidence=0.8,
        **tracker_kwargs
    ):
        """asdfasdfasdasdf"""

        self.model = model

        # Load the feature extraction network from checkpoints
        parent_dir = os.path.dirname(__file__)
        self.extractor = Extractor([64, 128])
        self.extractor.load_weights(
            os.path.join(parent_dir, "checkpoints/extractor.tf")
        )

        # Configure the SORT tracker
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, **tracker_kwargs)

    def _preprocess(self, img, size):
        """asdfasdfasdf"""
        img_in = tf.expand_dims(img_in, 0)
        img_in = tf.image.resize(img_in, (size, size))
        img_in = img_in / 255

    def _postprocess(self, output):
        return output

    def _inference(self, img, preprocess=None, postprocess=None):
        if preprocess:
            img_in = preprocess(img)

        else:
            img_in = self._preprocess(img)

        # Run the inference
        output = self.model.predict(img_in)

        # Postprocess the results according to user defined function to pack the output
        # into bboxes and scores
        if postprocess:
            bboxes, scores = postprocess(output)

        else:
            bboxes, scores = self._postprocess(output)

        return bboxes, scores

    def track(self, img, preprocess=None, postprocess=None, detections=None, tlbr=True):
        """asdfasdfsd"""
        # Preprocess the image according to user defined function if specified
        if detections:
            bboxes, scores = detections

        else:
            bboxes, scores = self._inference(img, preprocess, postprocess)

        detections = []
        for i in range(len(bboxes)):
            score = scores[i]

            # Skip if not confident enough
            if score < self.min_confidence:
                continue

            # Unpack the bbox
            bbox = bboxes[i]
            top, left, bottom, right = bbox

            # Extract feature vector
            patch = img[left:right, top:bottom]
            patch = tf.expand_dims(patch, 0)
            patch = tf.image.resize(img, [64, 128])
            feature = self.extractor.predict(patch)

            # Create detection
            detections.append(Detection(bbox, score, feature))

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        # Pack up the output
        bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            if tlbr:
                bbox = track.to_tlbr()

            else:
                bbox = track.to_tlwh()

            bboxes.append(bbox)

        return bboxes
