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
        nn_budget=100,
        max_cosine_distance=0.5,
        nms_max_overlap=0.3,
        min_confidence=0.8,
        **tracker_kwargs
    ):
        """asdfasdfasdasdf"""

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

    def track(self, img, bboxes, scores, tlbr=True, **kwargs):
        """Tracks"""

        detections = []
        for i in range(len(bboxes)):
            score = scores[i]

            # Skip if not confident enough
            if score < self.min_confidence:
                continue

            # Unpack the bbox
            bbox = bboxes[i]
            if tlbr:
                top, left, bottom, right = map(lambda x: int(x), bbox)

            else:
                top, left, width, height = map(lambda x: int(x), bbox)
                right = left + width
                bottom = top + height

            # Extract feature vector
            patch = img[left:right, top:bottom]
            patch = tf.expand_dims(patch, 0)

            # Check just in case
            if 0 in patch.shape:
                continue

            # Resize and normalize
            patch = tf.image.resize(patch, [64, 128])
            patch /= 255

            # Inference on the bbox
            # TODO: Faster to inference on all in a batch?
            feature = np.squeeze(self.extractor.predict(patch))

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
        track_ids = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            if tlbr:
                bbox = track.to_tlbr()

            else:
                bbox = track.to_tlwh()

            bboxes.append(bbox)
            track_ids.append(track.track_id)

        return bboxes, track_ids
