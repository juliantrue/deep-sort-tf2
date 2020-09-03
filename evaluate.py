import os
import cv2
import numpy as np
from tqdm import tqdm


from deepsort import DeepSortTracker
from deepsort.sort import Detection
from application_util import gather_sequence_info, create_detections


def draw_bboxes(img, bboxes, tlbr=True):
    for bbox in bboxes:
        if tlbr:
            x1, y1, x2, y2 = map(lambda x: int(x), bbox)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            x, y, w, h = map(lambda x: int(x), bbox)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def main():
    # Configuration
    draw = False
    root = "/MOT16/train/"
    min_detection_height = 0

    for seq in os.listdir(root):
        sequence_dir = os.path.join(root, seq)
        detection_file = os.path.join(sequence_dir, "det/det.txt")
        output_file = "results/{}.txt".format(seq)

        print("Running on sequence_dir: {}".format(seq))

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
            # TODO: Change the trackid from uuid back to numbers
            bboxes, track_ids = tracker.track(img, detections=detections, tlbr=False)

            if draw:
                img = draw_bboxes(img, bboxes, tlbr=False)
                cv2.imshow("View", img)
                cv2.waitKey(1)

            for i in range(len(bboxes)):
                results.append(
                    [
                        frame_idx,
                        track_ids[i],
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
