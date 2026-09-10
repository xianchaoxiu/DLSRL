#!/usr/bin/env python3
"""Export two Square W&B runs on a shared timestep grid."""

import bisect
import csv
from pathlib import Path

import wandb


PROJECT = "pfzhang911-shanghai-university/DSRL"
RUNS = [
    {
        "id": "wx7lgwfd",
        "label": "DLSRL",
    },
    {
        "id": "7r95y4lh",
        "label": "DSRL",
    },
]
REFERENCE_RUN_ID = "wx7lgwfd"
MAX_POINTS = 20

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "plots" / "square_eval_success_curve_points.csv"


def fetch_history(api, run_id):
    run = api.run(f"{PROJECT}/{run_id}")
    history = run.scan_history(
        keys=["eval/timesteps", "eval/success_rate"]
    )
    points = [
        (
            int(round(float(row["eval/timesteps"]))),
            float(row["eval/success_rate"]),
        )
        for row in history
        if row.get("eval/timesteps") is not None
        and row.get("eval/success_rate") is not None
    ]
    points.sort()
    if not points:
        raise RuntimeError(f"No evaluation history found for W&B run {run_id}")
    return points


def downsample_points(points, max_points):
    if max_points < 2:
        raise ValueError("MAX_POINTS must be at least 2")
    if len(points) <= max_points:
        return points

    last_index = len(points) - 1
    indices = [
        round(index * last_index / (max_points - 1))
        for index in range(max_points)
    ]
    return [points[index] for index in indices]


def interpolate(points, timestep):
    timesteps = [point[0] for point in points]
    index = bisect.bisect_left(timesteps, timestep)

    if index < len(points) and points[index][0] == timestep:
        return points[index][1], "observed"
    if index == 0 or index == len(points):
        raise ValueError(f"Timestep {timestep} is outside the interpolation range")

    left_timestep, left_value = points[index - 1]
    right_timestep, right_value = points[index]
    weight = (timestep - left_timestep) / (right_timestep - left_timestep)
    value = left_value + weight * (right_value - left_value)
    return value, "linear_interpolation"


def main():
    api = wandb.Api(timeout=60)
    histories = {
        run_info["id"]: fetch_history(api, run_info["id"])
        for run_info in RUNS
    }

    shared_end = min(points[-1][0] for points in histories.values())
    reference_points = [
        point
        for point in histories[REFERENCE_RUN_ID]
        if point[0] <= shared_end
    ]
    reference_points = downsample_points(reference_points, MAX_POINTS)
    common_timesteps = [point[0] for point in reference_points]

    aligned = {}
    for run_info in RUNS:
        run_id = run_info["id"]
        aligned[run_id] = [
            (timestep, *interpolate(histories[run_id], timestep))
            for timestep in common_timesteps
        ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "series",
            "run_id",
            "point_index",
            "timesteps",
            "success_rate",
            "value_type",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for run_info in RUNS:
            for point_index, (timestep, success_rate, value_type) in enumerate(
                aligned[run_info["id"]], start=1
            ):
                writer.writerow(
                    {
                        "series": run_info["label"],
                        "run_id": run_info["id"],
                        "point_index": point_index,
                        "timesteps": timestep,
                        "success_rate": round(success_rate, 6),
                        "value_type": value_type,
                    }
                )

    print(f"Shared range: 0 to {shared_end}")
    print(f"Common timesteps ({len(common_timesteps)}): {common_timesteps}")
    for run_info in RUNS:
        counts = {}
        for _, _, value_type in aligned[run_info["id"]]:
            counts[value_type] = counts.get(value_type, 0) + 1
        print(f"{run_info['label']}: {counts}")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
