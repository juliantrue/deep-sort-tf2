import os
import subprocess
import pandas as pd


results_dir = "results"
exclude_dirs = ["first_test", "second_test"]

results = {}
percent_idxs = [0, 1, 2, 3, 4, 13]

for folder in os.listdir(results_dir):
    if folder in exclude_dirs:
        continue

    folder_path = os.path.join(results_dir, folder)

    values = folder.split("_")
    keys = [
        "nn_budget",
        "max_cosine_distance",
        "max_nms_overlap",
        "min_confidence",
        "max_iou_distance",
        "n_init",
    ]
    params = {k: v for k, v in zip(keys, values)}

    print(params)
    cmd = f"python3 -u -m motmetrics.apps.eval_motchallenge /MOT16/train ./results/{folder}".split(
        " "
    )
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8")
    result = result.split("\n")
    keys = [k for k in result[0].split(" ") if not k == ""]
    vals = [v for v in result[1].split(" ") if not v == ""][1:]
    values = []
    for i, v in enumerate(vals):
        if i in percent_idxs:
            values.append(v[:-1])

        else:
            values.append(v)

    if results == {}:
        results = {k: [] for k in keys}

    results = {k: results[k] + [v] for k, v in zip(results.keys(), values)}


df = pd.DataFrame(results)
