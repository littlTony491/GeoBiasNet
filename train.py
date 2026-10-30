"""Train only the final original-reshape model; no ablation selection."""

import json
from pathlib import Path

import torch
import numpy as np

from runtime import (
    VARIANT, build_model, load_checkpoint, make_loader, parse_options, run_epoch,
    save_checkpoint, save_json, seed_everything, selected_actions,
)


def main(argv=None):
    opt = parse_options(argv)
    seed_everything(opt.seed)
    output = Path(opt.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not opt.resume and any((output/name).exists() for name in ("last.pt", "best.pt", "history.jsonl")):
        raise FileExistsError("use a new output directory or --resume latest")
    resume_path = output/"last.pt" if opt.resume == "latest" else Path(opt.resume) if opt.resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    print(f"variant={VARIANT} dataset=h36m device={opt.device} "
          f"input_frames={opt.input_n} output_frames={opt.output_n} validation_split={opt.validation_split}", flush=True)
    train_loader = make_loader(opt, split=0, training=True)
    val_loader = make_loader(opt, split=opt.validation_split, shuffle=True)
    test_loaders = {action: make_loader(opt, split=2, action=action) for action in selected_actions(opt)}
    model = build_model(opt)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    completed, best_error, learning_rate = 0, float("inf"), opt.lr
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, opt, optimizer)
        completed, best_error = checkpoint["epoch"], checkpoint["best_err"]
        learning_rate = checkpoint["lr"]
        print(f"resumed={resume_path} completed_epochs={completed}", flush=True)
    save_json(output/"config.json", {"model_variant": VARIANT, **vars(opt)})
    for epoch in range(completed + 1, opt.epochs + 1):
        # Archived training decays before training, including the first epoch.
        learning_rate *= opt.lr_decay
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train = run_epoch(model, train_loader, opt, optimizer, phase="train")
        val = run_epoch(model, val_loader, opt, phase="validation")
        test_actions = {action: run_epoch(model, loader, opt, phase=f"test:{action}")
                        for action, loader in test_loaders.items()}
        test_mpjpe = np.asarray([metrics["mpjpe"] for metrics in test_actions.values()]).mean(axis=0)
        is_best = val["loss"] < best_error
        best_error = min(best_error, val["loss"])
        record = {"model_variant": VARIANT, "epoch": epoch, "learning_rate": learning_rate,
                  "train": train, "val": val, "best_val_loss": best_error,
                  "test": {"actions": test_actions, "mpjpe": test_mpjpe.tolist(),
                           "loss": float(test_mpjpe.mean()), "aggregation": "mean over actions"},
                  "metric": "32-joint future MPJPE in millimetres; sample mean"}
        with (output/"history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_json(output/f"metrics_epoch_{epoch:04d}.json", record)
        save_checkpoint(output/"last.pt", model, optimizer, epoch, best_error, record, opt)
        save_checkpoint(output/f"epoch_{epoch:04d}.pt", model, optimizer, epoch, best_error, record, opt)
        if is_best:
            save_checkpoint(output/"best.pt", model, optimizer, epoch, best_error, record, opt)
        print(f"epoch={epoch}/{opt.epochs} lr={learning_rate:.8g} "
              f"train_mpjpe_32j={train['loss']:.5f} val_mpjpe_32j={val['loss']:.5f} "
              f"test_mpjpe_action_mean={test_mpjpe.mean():.5f} "
              f"objective_sum5={train['objective']:.5f} best_val={best_error:.5f}", flush=True)
    print(f"best_checkpoint={output/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
