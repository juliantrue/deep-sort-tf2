import os
import time
import cv2
import multiprocessing
import numpy as np
from tqdm import tqdm


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


def evaluate(
    nn_budget=100,
    max_cosine_distance=0.5,
    nms_max_overlap=0.3,
    min_confidence=0.8,
    max_iou_distance=0.7,
    n_init=3,
):
    from deepsort import DeepSortTracker

    # Configuration
    draw = False
    mot16dets = False

    # WARNING: only run on the "train" split if intending to us MOT. Test has no GT
    split = "train"
    root = f"/MOT16/{split}/"

    if mot16dets:
        detections_folder = root

    else:
        detections_folder = f"data/MOT16_POI_{split}"

    min_detection_height = 0

    print(
        f"Perfoming evaluatation with parameters:"
        f"\nnn_budget: {nn_budget}"
        f"\nmax_cosine_distance: {max_cosine_distance}"
        f"\nnms_max_overlap: {nms_max_overlap}"
        f"\nmin_confidence: {min_confidence}"
        f"\nmax_iou_distance: {max_iou_distance}"
        f"\nn_init: {n_init}"
    )

    results_folder = "results"

    for seq in os.listdir(root):
        sequence_dir = os.path.join(root, seq)
        if mot16dets:
            detection_file = os.path.join(detections_folder, seq + "/det/det.txt")

        else:
            detection_file = os.path.join(detections_folder, seq + ".npy")
        print("Running on sequence_dir: {}".format(seq))

        # Initialize the tracker
        tracker = DeepSortTracker(
            nn_budget,
            max_cosine_distance,
            nms_max_overlap,
            min_confidence,
            max_iou_distance,
            n_init,
        )
        seq_info = gather_sequence_info(sequence_dir, detection_file)

        # Evaulate on MOT
        results = []
        for frame_idx in tqdm(
            range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1)
        ):

            # Get the image
            img = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

            # Get the detections
            bboxes, scores = create_detections(
                seq_info["detections"], frame_idx, min_detection_height
            )

            # Run the tracker
            last = time.time()
            bboxes, track_ids = tracker.track(img, bboxes, scores, tlbr=False)
            now = time.time()
            # print(f"{(now-last)*1000}ms")

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

        output_file = os.path.join(results_folder, f"{seq}.txt")
        f = open(output_file, "w")
        for row in results:
            print(
                "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                % (row[0], row[1], row[2], row[3], row[4], row[5]),
                file=f,
            )


if __name__ == "__main__":
    # TODO: Run each of these
    evaluate(128, 0.3, 1.0, 0.5, 0.7, 2)  # w custom model
    # evaluate(128, 0.3, 1.0, 0.3, 0.7, 3) # original model
