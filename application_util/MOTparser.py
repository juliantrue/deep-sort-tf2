import os
import cv2
import numpy as np


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).
    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.
    Returns
    -------
    Dict
        A dictionary of the following sequence information:
        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.
    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
    }
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.loadtxt(detection_file, delimiter=",")
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=",")

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split("=") for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2
            )

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.
    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.
    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    bboxes = []
    scores = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue

        else:
            bbox[:2] += bbox[2:] / 2
            bbox[2] /= bbox[3]
            bboxes.append(bbox)
            scores.append(confidence)

    return (bboxes, scores)
