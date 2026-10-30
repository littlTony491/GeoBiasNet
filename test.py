"""Test the final model with the published 32-joint, action-mean protocol."""

import csv
from pathlib import Path

import numpy as np

from runtime import (
    VARIANT, build_model, load_checkpoint, make_loader, parse_options, run_epoch,
    save_json, seed_everything, selected_actions,
)


def main(argv=None):
    opt = parse_options(argv)
    seed_everything(opt.seed)
    output = Path(opt.output_dir)
    path = Path(opt.checkpoint) if opt.checkpoint else output/"best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    model = build_model(opt)
    checkpoint = load_checkpoint(path, model, opt)
    actions = {}
    for action in selected_actions(opt):
        loader = make_loader(opt, split=2, action=action)
        actions[action] = run_epoch(model, loader, opt, phase=f"test:{action}")
        print(f"action={action} samples={actions[action]['samples']} "
              f"mpjpe_32j={actions[action]['loss']:.5f}", flush=True)
    # Equal weight per action, as in the released test script.
    average = np.asarray([value["mpjpe"] for value in actions.values()]).mean(axis=0)
    result = {"model_variant": VARIANT, "checkpoint": str(path.resolve()),
              "checkpoint_epoch": checkpoint.get("epoch"), "input_n": opt.input_n,
              "output_n": opt.output_n, "milliseconds": [(i+1)*40 for i in range(opt.output_n)],
              "metric": "32-joint MPJPE in millimetres; mean over actions",
              "full_sequence_input": True, "reshape_preserved": True,
              "actions": actions, "average": {"mpjpe": average.tolist(), "loss": float(average.mean())}}
    save_json(output/"test_metrics.json", result)
    with (output/"test_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action", *[f"{(i+1)*40}ms" for i in range(opt.output_n)]])
        for action, value in actions.items():
            writer.writerow([action, *value["mpjpe"]])
        writer.writerow(["avg", *average.tolist()])
    print(f"checkpoint={path} action_mean_mpjpe_32j={average.mean():.5f}", flush=True)
    print(f"metrics={output/'test_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
