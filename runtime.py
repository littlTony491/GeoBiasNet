"""H3.6M training/testing support for the original, intentionally leaky model.

The five-output objective and 32-joint metric follow archived training code.
The complete history+future window is deliberately passed into the model.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from model import MultiStageModel_4
from utils.h36motion3d import Datasets


ROOT = Path(__file__).resolve().parent
VARIANT = "geobiasnet-original-reshape-v1"
ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions", "greeting",
    "phoning", "posing", "purchases", "sitting", "sittingdown", "takingphoto",
    "waiting", "walkingdog", "walkingtogether",
]
DIM_USED = np.array([
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92,
])
IGNORED_JOINTS = np.array([16, 20, 23, 24, 28, 31])
EQUAL_JOINTS = np.array([13, 19, 22, 13, 27, 30])
INDEX_TO_IGNORE = np.concatenate([IGNORED_JOINTS * 3 + i for i in range(3)])
INDEX_TO_EQUAL = np.concatenate([EQUAL_JOINTS * 3 + i for i in range(3)])


def parse_options(argv=None):
    local_data = ROOT / "dataset" / "h3.6m"
    sibling_data = ROOT.parent / "BARNet" / "dataset" / "h3.6m"
    default_data = local_data if local_data.is_dir() or not sibling_data.is_dir() else sibling_data
    parser = argparse.ArgumentParser(description="Original GeoBiasNet reshape reproduction, H3.6M only")
    parser.add_argument("--data-dir", default=str(default_data))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-n", type=int, default=10)
    parser.add_argument("--output-n", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--drop-out", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=None,
                        help="default: 0.1 ** (1/epochs), applied before each epoch as in archived training")
    parser.add_argument("--max-norm", type=float, default=10000)
    parser.add_argument("--seed", type=int, default=334)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-rate", type=int, default=1)
    parser.add_argument("--test-sample-num", type=int, default=-1)
    parser.add_argument("--actions", nargs="+", choices=["all", *ACTIONS], default=["all"])
    parser.add_argument("--validation-split", type=int, choices=[1, 2], default=2,
                        help="2=S5, preserves archived validation; 1=S11, explicitly chooses a separate subject")
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint", default=None)
    opt = parser.parse_args(argv)
    if opt.input_n != 10 or opt.output_n < 1 or opt.d_model < 4 or opt.d_model % 4:
        parser.error("requires input_n=10, output_n>0 and d_model a positive multiple of 4")
    if opt.batch_size < 2 or opt.test_batch_size < 1 or opt.epochs < 1:
        parser.error("batch_size>=2, test_batch_size>=1 and epochs>=1 are required")
    if opt.skip_rate < 1 or opt.max_batches == 0 or opt.max_batches < -1 or opt.log_every < 0:
        parser.error("invalid skip-rate, max-batches or log-every")
    if "all" in opt.actions and len(opt.actions) != 1:
        parser.error("use --actions all or a list of named actions")
    if opt.device == "auto":
        opt.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    opt.cuda_idx = opt.device
    opt.kernel_size, opt.in_features, opt.num_stage = 10, 66, 1
    opt.train_output_n = opt.output_n
    opt.dct_n = opt.input_n + opt.output_n
    opt.lr_decay = opt.lr_decay if opt.lr_decay is not None else 0.1 ** (1 / opt.epochs)
    opt.data_dir = str(Path(opt.data_dir).resolve())
    opt.output_dir = str(Path(opt.output_dir).resolve()) if opt.output_dir else str(
        ROOT / "checkpoints" / f"h36m_original_reshape_10in{opt.output_n}out"
    )
    return opt


def selected_actions(opt):
    return ACTIONS if opt.actions == ["all"] else opt.actions


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(opt, split, training=False, action=None, shuffle=None):
    if action is not None:
        dataset = Datasets(opt=opt, split=split, actions=action)
    elif opt.actions == ["all"]:
        dataset = Datasets(opt=opt, split=split)
    else:
        dataset = ConcatDataset([Datasets(opt=opt, split=split, actions=a) for a in opt.actions])
    if not len(dataset):
        raise RuntimeError(f"empty dataset: {opt.data_dir}, split={split}")
    return DataLoader(
        dataset, batch_size=opt.batch_size if training else opt.test_batch_size,
        shuffle=training if shuffle is None else shuffle,
        num_workers=opt.num_workers, pin_memory=opt.device.startswith("cuda"),
        drop_last=False,
    )


def build_model(opt):
    return MultiStageModel_4(opt).to(opt.device)


def full_sequence_losses(outputs, target):
    batch, frames, _ = target.shape
    target_joints = target.reshape(batch, frames, 22, 3)
    losses = [torch.mean(torch.norm(out.reshape(batch, frames, 22, 3) - target_joints, dim=3))
              for out in outputs]
    # Match archived addition order, including all observed and future frames.
    objective = losses[0] + (losses[1] + losses[2]) + (losses[3] + losses[4])
    return objective, losses[0]


def legacy_32_joint_distances(prediction, batch, opt):
    # Preserve published testing: fill active joints, copy equivalent joints,
    # and retain the original future GT for the remaining coordinates.
    out = batch.clone()[:, opt.input_n:opt.input_n + opt.output_n]
    out[:, :, DIM_USED] = prediction[:, opt.input_n:]
    out[:, :, INDEX_TO_IGNORE] = out[:, :, INDEX_TO_EQUAL]
    out = out.reshape(batch.shape[0], opt.output_n, 32, 3)
    target = batch.reshape(batch.shape[0], opt.dct_n, 32, 3)[:, opt.input_n:]
    return torch.norm(target - out, dim=3)


def run_epoch(model, loader, opt, optimizer=None, phase="validation"):
    training = optimizer is not None
    model.train(training)
    count, batches = 0, 0
    objective_sum = final_loss_sum = mean_error_sum = 0
    # Published testing accumulates per-frame float32 batch sums in a NumPy
    # float64 array. Keep that order and precision across multiple batches.
    errors = np.zeros(opt.output_n)
    started = time.monotonic()
    with (nullcontext() if training else torch.no_grad()):
        for index, batch in enumerate(loader):
            if opt.max_batches > 0 and index >= opt.max_batches:
                break
            if training and batch.shape[0] == 1:
                continue  # Historical BatchNorm tail handling.
            batch = batch.float().to(opt.device)
            # INTENTIONAL: do not truncate to observed history here.
            full_input = batch[:, :, DIM_USED].clone()
            outputs = model(full_input)
            if training:
                objective, final_loss = full_sequence_losses(outputs, full_input)
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=opt.max_norm)
                optimizer.step()
                objective_sum += objective.detach().cpu().numpy() * len(batch)
                final_loss_sum += final_loss.detach().cpu().numpy() * len(batch)
            with torch.no_grad():
                distances = legacy_32_joint_distances(outputs[0], batch, opt)
                errors += distances.mean(dim=2).sum(dim=0).cpu().numpy()
                mean_error_sum += distances.mean().cpu().numpy() * len(batch)
            count += len(batch)
            batches += 1
            if opt.log_every and (batches == 1 or batches % opt.log_every == 0):
                elapsed = time.monotonic() - started
                print(f"phase={phase} batch={batches} samples={count} "
                      f"running_mpjpe_32j={float(mean_error_sum/count):.5f} elapsed_s={elapsed:.1f}", flush=True)
    if not count:
        raise RuntimeError("no usable batches; training needs at least two samples")
    result = {"mpjpe": (errors/count).tolist(), "loss": float(mean_error_sum/count),
              "samples": count, "batches": batches}
    if training:
        result.update(objective=float(objective_sum/count), final_stage_loss=float(final_loss_sum/count))
    return result


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, epoch, best_error, record, opt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_variant": VARIANT, "epoch": epoch, "err": record["val"]["loss"],
               "best_err": best_error, "lr": optimizer.param_groups[0]["lr"],
               "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng_state": rng_state(), "args": vars(opt), "metrics": record}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, opt, optimizer=None):
    # Use only trusted local/published checkpoints: full training state uses pickle.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    marker = checkpoint.get("model_variant", checkpoint.get("model_format"))
    if marker is not None and marker != VARIANT:
        raise ValueError(f"checkpoint belongs to another implementation: {marker}")
    if optimizer is not None and marker != VARIANT:
        raise ValueError("resume requires this reproduction version's full training checkpoint")
    saved = checkpoint.get("args", {})
    fields = ["input_n", "output_n", "dct_n", "d_model", "in_features"]
    if optimizer is not None:
        fields += ["drop_out", "lr", "lr_decay", "max_norm", "validation_split", "actions"]
    for field in fields:
        if field in saved and saved[field] != getattr(opt, field):
            raise ValueError(f"checkpoint mismatch for {field}: {saved[field]!r} vs {getattr(opt, field)!r}")
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng_state"])
    return checkpoint
