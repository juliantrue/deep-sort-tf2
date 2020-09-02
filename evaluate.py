import os
import cv2
import numpy as np
from tqdm import tqdm


from deepsort import DeepSortTracker
from deepsort.sort import Detection
from application_util import gather_sequence_info, create_detections


def main():
    # Configuration
    sequence_dir = "/MOT16/train/MOT16-02"
    detection_file = "/MOT16/train/MOT16-02/det/det.txt"
    output_file = "output.txt"
    min_detection_height = 0

    # Initialize the tracker
    tracker = DeepSortTracker()
    seq_info = gather_sequence_info(sequence_dir, detection_file)

    # Evaulate on MOT
    results = []
    for frame_idx in tqdm(
        range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1)
    ):

        # Get the image
        img = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        # Get the detections
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height
        )

        # Run the tracker
        bboxes, track_ids = tracker.track(img, detections=detections)

        for i in range(len(bboxes)):
            results.append(
                [
                    frame_idx,
                    track_ids,
                    bboxes[i][0],
                    bboxes[i][1],
                    bboxes[i][2],
                    bboxes[i][3],
                ]
            )

    # Store results.
    f = open(output_file, "w")
    for row in results:
        print(
            "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
            % (row[0], row[1], row[2], row[3], row[4], row[5]),
            file=f,
        )


if __name__ == "__main__":
    main()
