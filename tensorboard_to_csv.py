import argparse
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, TENSORS


def run(args):
    src_dir = args.src
    dst_dir = args.dst
    batch_size = args.aggregate
    cpu_mem = args.cm

    event_acc = EventAccumulator(src_dir, size_guidance={TENSORS: 2_000_000}).Reload()
    tags = event_acc.Tags()["tensors"]

    value_dict = {}
    eval_value_dict = {}

    for tag in tags:
        if cpu_mem:
            if not "CPU" in tag and not "Memory" in tag:
                continue
        else:
            if "CPU" in tag or "Memory" in tag:
                continue
                
        metrics = np.array([
            float(tf.make_ndarray(event.tensor_proto))
            for event in event_acc.Tensors(tag)
        ])

        metrics = metrics.astype(float)

        remainder = len(metrics) % batch_size
        pad_width = batch_size - remainder if remainder > 0 else 0
        metrics = np.pad(
            array=metrics, 
            pad_width=(0, pad_width),
            mode="constant",
            constant_values=np.nan,
        )
        metrics = np.reshape(metrics, (-1, batch_size))
        metrics = np.sum(metrics, axis=1) if "CPU" in tag else np.nanmean(metrics, axis=1)

        if tag.startswith("eval"):
            eval_value_dict[tag] = metrics
        else:
            value_dict[tag] = metrics

    print("value_dict")
    for key, value in value_dict.items():
        print(f"{key=}, {len(value)=}")

    print("eval_value_dict")
    for key, value in eval_value_dict.items():
        print(f"{key=}, {len(value)=}")

    file_name = ("cm_" if cpu_mem else "") + os.path.basename(src_dir)
    file_path = os.path.join(dst_dir, f"{file_name}.csv")
    eval_file_path = os.path.join(dst_dir, f"{file_name}_eval.csv")

    os.makedirs(dst_dir, exist_ok=True)

    pd.DataFrame(value_dict).to_csv(file_path, index=False)
    pd.DataFrame(eval_value_dict).to_csv(eval_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to directory with tensorboard logs for a single run",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path to destination directory where to save csv",
    )
    parser.add_argument(
        "--aggregate",
        type=int,
        required=False,
        default=1,
        help="Number of consecutive metrics to aggregate (average or sum). Defaults to 1 which implies no aggregation"
    )
    parser.add_argument(
        "--cm",
        type=bool,
        required=False,
        default=False,
        help="True to extract CPU and Memory metrics, False to extract SLO compliance metrics"
    )
    args = parser.parse_args()

    run(args)
