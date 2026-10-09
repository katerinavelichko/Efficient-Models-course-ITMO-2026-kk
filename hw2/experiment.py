import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from data import make_datasets, make_loader, subset_loader
from models import ARCHS, build_model, count_parameters
from plots import make_plots
from selection import (
    forgetting_selection_score,
    kcenter_scores,
    pearson,
    random_subset,
    select_top,
    select_top_excluding,
    spearman,
    spearman_masked,
)
from train_loop import collect_features, predict_uncertainty, scaled_epoch_lrs, train_model


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_cuda_fp32() -> None:
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if setter is not None:
        setter("highest")


def default_workers() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        st = os.statvfs("/dev/shm")
        free = st.f_frsize * st.f_bavail
    except OSError:
        return 0
    if free < 2 * 1024 ** 3:
        log(f"/dev/shm is {free / 1024 ** 2:.0f} MiB, using num_workers=0")
        return 0
    return 2


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def append_csv(path: str, row: dict, fieldnames: list) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{float(value):.6f}"


def never_learned_fraction(ever_correct: np.ndarray) -> float:
    return float((~np.asarray(ever_correct, dtype=bool)).mean())


def summarize_lrs(lrs) -> str:
    parts = []
    for lr in lrs:
        if not parts or parts[-1][0] != lr:
            parts.append([lr, 1])
        else:
            parts[-1][1] += 1
    return ", ".join(f"{count} x {lr:g}" for lr, count in parts)


def active_budgets(n: int):
    initial = min(n // 2, max(8, int(round(0.02 * n))))
    b10 = min(n - 2, max(initial + 1, int(round(0.10 * n))))
    b20 = min(n, max(b10 + 1, int(round(0.20 * n))))
    return initial, b10, b20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--snapshot-epoch", type=int, default=10)
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--subsample", type=int, default=2000, help="points used only for greedy k-centers")
    p.add_argument("--quick", action="store_true", help="synthetic data, 1 epoch, for a smoke test")
    p.add_argument("--skip-active", action="store_true")
    p.add_argument("--skip-coreset", action="store_true")
    p.add_argument("--skip-mid", action="store_true", help="do not train PreAct-ResNet-20")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.snapshot_epoch = 1
        args.subsample = 48
        args.batch_size = 32
        args.eval_batch_size = 64
        args.synthetic = True
        args.synthetic_n = 128
        log("quick: synthetic data, 1 epoch (overrides --epochs)")
    else:
        args.synthetic = False
        args.synthetic_n = 0
    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be in (0, 1]")
    if args.snapshot_epoch > args.epochs:
        args.snapshot_epoch = args.epochs
    if args.snapshot_epoch < 1:
        args.snapshot_epoch = 1
    if args.num_workers < 0:
        args.num_workers = default_workers()

    os.makedirs(args.out, exist_ok=True)
    identity = {
        "epochs": args.epochs,
        "snapshot_epoch": args.snapshot_epoch,
        "fraction": args.fraction,
        "seed": args.seed,
        "synthetic": args.synthetic,
        "synthetic_n": args.synthetic_n,
        "batch_size": args.batch_size,
        "subsample": args.subsample,
    }
    config_path = os.path.join(args.out, "config.json")
    if os.path.exists(config_path):
        previous = read_json(config_path)
        if previous != identity:
            raise SystemExit(
                f"{config_path} does not match this run.\n"
                f"  saved: {previous}\n  now:   {identity}\n"
                f"Delete {args.out} or pass the same epochs/seed/fraction."
            )
    else:
        dump_json(config_path, identity)

    configure_cuda_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    if device.type != "cuda" and not args.quick:
        log("WARNING: no CUDA device. A full CIFAR run on CPU will take many hours.")
    log(f"device: {gpu}  torch {torch.__version__}")
    log("lr schedule: " + summarize_lrs(scaled_epoch_lrs(args.epochs)))

    train_aug, train_eval, test_ds = make_datasets(
        args.data, args.synthetic, args.synthetic_n if args.synthetic else 128, args.seed
    )
    n = len(train_aug)
    num_classes = int(train_aug.labels.max()) + 1
    workers = args.num_workers
    batch_size = min(args.batch_size, n)
    eval_bs = min(args.eval_batch_size, n)
    train_eval_loader = make_loader(train_eval, eval_bs, shuffle=False, workers=workers)
    test_loader = make_loader(test_ds, eval_bs, shuffle=False, workers=workers)
    log(f"train examples: {n}   test examples: {len(test_ds)}   classes: {num_classes}")

    params = {arch: count_parameters(build_model(arch, num_classes)) for arch in ARCHS}
    for arch, count in params.items():
        log(f"  {arch}: {count / 1e6:.3f}M parameters")

    sub_n = min(args.subsample, n)
    start_n = min(max(4, sub_n // 20), sub_n // 2)
    sub_path = os.path.join(args.out, "subsample_indices.npy")
    start_path = os.path.join(args.out, "kcenter_start.npy")
    if os.path.exists(sub_path):
        subsample = np.load(sub_path)
        k_start = np.load(start_path)
    else:
        rng = np.random.default_rng(args.seed)
        subsample = np.sort(rng.choice(n, size=sub_n, replace=False).astype(np.int64))
        k_start = rng.choice(sub_n, size=start_n, replace=False).astype(np.int64)
        np.save(sub_path, subsample)
        np.save(start_path, k_start)
    log(f"k-centers subsample: {len(subsample)} points, initial pool {len(k_start)}")

    specs = [
        ("preact8", args.seed, "preact8_s0"),
        ("preact8", args.seed + 1, "preact8_s1"),
    ]
    if not args.skip_mid:
        specs.append(("preact20", args.seed, "preact20_s0"))
    specs.append(("preact56", args.seed, "preact56_s0"))

    rank_dir = os.path.join(args.out, "rankings")
    os.makedirs(rank_dir, exist_ok=True)

    def fit(arch, seed, indices, *, track, snapshot, uncertainty, features, eval_test, prefix, eval_every, log_every):
        set_seed(seed)
        model = build_model(arch, num_classes).to(device)
        if indices is None:
            loader = make_loader(train_aug, batch_size, True, workers, seed)
        else:
            loader = subset_loader(train_aug, indices, batch_size, True, workers, seed)
        extra = {}

        def on_snap(current):
            stats = predict_uncertainty(current, train_eval_loader, n, device)
            extra["entropy"] = stats["entropy"]
            extra["clean_acc"] = stats["accuracy"]
            extra["features"] = collect_features(current, train_eval, subsample, device, eval_bs, workers)
            return {}

        result = train_model(
            model,
            loader,
            device,
            scaled_epoch_lrs(args.epochs),
            test_loader=test_loader if eval_test else None,
            eval_every=eval_every,
            n_train=n if track else None,
            track_forgetting=track,
            on_snapshot=on_snap if snapshot else None,
            snapshot_epoch=args.snapshot_epoch if snapshot else None,
            log_prefix=prefix,
            log_every=log_every,
        )
        result["snapshot_extra"] = extra
        if track or uncertainty:
            stats = predict_uncertainty(result["model"], train_eval_loader, n, device)
            result["entropy"] = stats["entropy"]
            result["least_confidence"] = stats["least_confidence"]
            result["clean_acc"] = stats["accuracy"]
        if track or features:
            result["features"] = collect_features(result["model"], train_eval, subsample, device, eval_bs, workers)
        del result["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    for arch, seed, key in specs:
        meta_path = os.path.join(rank_dir, key + ".json")
        npz_path = os.path.join(rank_dir, key + ".npz")
        if os.path.exists(meta_path) and os.path.exists(npz_path):
            log(f"resume ranking {key}")
            continue
        log(f"=== train {key} ({arch}, seed {seed}) on all {n} examples ===")
        result = fit(
            arch,
            seed,
            None,
            track=True,
            snapshot=(key == "preact56_s0"),
            uncertainty=False,
            features=False,
            eval_test=True,
            prefix=f"[{key}]",
            eval_every=1,
            log_every=1,
        )
        arrays = {
            "forgetting": result["forgetting"],
            "ever_correct": result["ever_correct"],
            "entropy": result["entropy"],
            "features": result["features"],
        }
        meta = {
            "arch": arch,
            "seed": seed,
            "test_acc": float(result["test_acc"]),
            "seconds": float(result["seconds"]),
            "clean_acc": float(result["clean_acc"]),
            "never_learned": never_learned_fraction(result["ever_correct"]),
            "params": params[arch],
            "has_early": False,
        }
        snap = result["snapshot"]
        extra = result["snapshot_extra"]
        if snap is not None and "entropy" in extra:
            arrays["forgetting_early"] = snap["forgetting"]
            arrays["ever_correct_early"] = snap["ever_correct"]
            arrays["entropy_early"] = extra["entropy"]
            arrays["features_early"] = extra["features"]
            meta["has_early"] = True
            meta["test_acc_early"] = float(snap["test_acc"])
            meta["never_learned_early"] = never_learned_fraction(snap["ever_correct"])
            meta["clean_acc_early"] = float(extra["clean_acc"])
        if key == "preact56_s0":
            hist_path = os.path.join(args.out, "history_preact56.csv")
            with open(hist_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "lr", "loss", "aug_acc", "test_acc", "epoch_seconds"]
                )
                writer.writeheader()
                writer.writerows(result["history"])
        np.savez_compressed(npz_path, **arrays)
        dump_json(meta_path, meta)
        log(
            f"done {key}: test error {100 * (1 - meta['test_acc']):.2f}%  "
            f"never-learned {100 * meta['never_learned']:.1f}%  {meta['seconds']:.0f}s"
        )
        if meta["never_learned"] > 0.30 and not args.quick:
            log("WARNING: >30% of examples were never correct. Forgetting ranks will be noisy; raise --epochs.")

    store = {}
    for _, _, key in specs:
        meta = read_json(os.path.join(rank_dir, key + ".json"))
        arrays = np.load(os.path.join(rank_dir, key + ".npz"))
        store[key] = {"meta": meta, **{name: arrays[name] for name in arrays.files}}
    if store["preact56_s0"]["meta"].get("has_early"):
        parent = store["preact56_s0"]
        store["preact56_early"] = {
            "forgetting": parent["forgetting_early"],
            "ever_correct": parent["ever_correct_early"],
            "entropy": parent["entropy_early"],
            "features": parent["features_early"],
            "meta": {
                "test_acc": parent["meta"].get("test_acc_early", float("nan")),
                "never_learned": parent["meta"].get("never_learned_early", float("nan")),
                "seconds": 0.0,
            },
        }

    log(f"greedy k-centers on {len(subsample)} features")
    t_kc = time.perf_counter()
    for key, item in store.items():
        item["kcenter"] = kcenter_scores(item["features"], k_start)
    log(f"k-centers done in {time.perf_counter() - t_kc:.1f}s")

    pair_list = [("R8 / R8", "preact8_s0", "preact8_s1"), ("R8 / R56", "preact8_s0", "preact56_s0")]
    if "preact20_s0" in store:
        pair_list[1:1] = [("R8 / R20", "preact8_s0", "preact20_s0")]
        pair_list.append(("R20 / R56", "preact20_s0", "preact56_s0"))
    if "preact56_early" in store:
        pair_list.append(("R56 early / R56", "preact56_early", "preact56_s0"))

    corr_fields = ["pair", "metric", "spearman", "pearson"]
    corr_path = os.path.join(args.out, "correlations.csv")
    corr_rows = []
    for label, a, b in pair_list:
        sa = forgetting_selection_score(store[a]["forgetting"], store[a]["ever_correct"])
        sb = forgetting_selection_score(store[b]["forgetting"], store[b]["ever_correct"])
        corr_rows.append({
            "pair": label,
            "metric": "forgetting",
            "spearman": fmt(spearman(sa, sb)),
            "pearson": fmt(pearson(store[a]["forgetting"].astype(np.float64), store[b]["forgetting"].astype(np.float64))),
        })
        corr_rows.append({
            "pair": label,
            "metric": "entropy",
            "spearman": fmt(spearman(store[a]["entropy"], store[b]["entropy"])),
            "pearson": fmt(pearson(store[a]["entropy"], store[b]["entropy"])),
        })
        ka, kb = store[a]["kcenter"], store[b]["kcenter"]
        mask = np.isfinite(ka) & np.isfinite(kb)
        corr_rows.append({
            "pair": label,
            "metric": "k-centers",
            "spearman": fmt(spearman_masked(ka, kb)),
            "pearson": fmt(pearson(ka[mask], kb[mask]) if mask.sum() >= 3 else float("nan")),
        })
    with open(corr_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=corr_fields)
        writer.writeheader()
        writer.writerows(corr_rows)
    for row in corr_rows:
        log(f"  Spearman {row['metric']:12s} {row['pair']:18s} {row['spearman']}")

    coreset_fields = ["method", "fraction", "n_subset", "test_acc", "test_error", "train_seconds"]
    coreset_path = os.path.join(args.out, "coreset.csv")
    if not args.skip_coreset:
        done = {row["method"] for row in read_csv(coreset_path)}
        m = int(round(args.fraction * n))
        os.makedirs(os.path.join(args.out, "subsets"), exist_ok=True)
        if "full" not in done:
            acc = float(store["preact56_s0"]["meta"]["test_acc"])
            append_csv(coreset_path, {
                "method": "full",
                "fraction": fmt(1.0),
                "n_subset": n,
                "test_acc": fmt(acc),
                "test_error": fmt(1.0 - acc),
                "train_seconds": fmt(float(store["preact56_s0"]["meta"]["seconds"])),
            }, coreset_fields)
        jobs = [
            ("random", None),
            ("forget_preact8", forgetting_selection_score(store["preact8_s0"]["forgetting"], store["preact8_s0"]["ever_correct"])),
            ("forget_preact56", forgetting_selection_score(store["preact56_s0"]["forgetting"], store["preact56_s0"]["ever_correct"])),
        ]
        if "preact56_early" in store:
            jobs.append((
                "forget_preact56_early",
                forgetting_selection_score(store["preact56_early"]["forgetting"], store["preact56_early"]["ever_correct"]),
            ))
        jobs.append(("entropy_preact8", store["preact8_s0"]["entropy"]))
        for i, (method, scores) in enumerate(jobs):
            if method in done:
                log(f"resume coreset {method}")
                continue
            if method == "random":
                chosen = random_subset(n, m, args.seed + 11)
            else:
                chosen = select_top(scores, m, args.seed + 11)
            np.save(os.path.join(args.out, "subsets", method + ".npy"), chosen)
            log(f"=== coreset {method}: retrain preact56 on {len(chosen)} / {n} ===")
            result = fit(
                "preact56",
                args.seed + 100 + i,
                chosen,
                track=False,
                snapshot=False,
                uncertainty=False,
                features=False,
                eval_test=True,
                prefix=f"[coreset {method}]",
                eval_every=0,
                log_every=5,
            )
            acc = float(result["test_acc"])
            append_csv(coreset_path, {
                "method": method,
                "fraction": fmt(args.fraction),
                "n_subset": int(len(chosen)),
                "test_acc": fmt(acc),
                "test_error": fmt(1.0 - acc),
                "train_seconds": fmt(float(result["seconds"])),
            }, coreset_fields)
            log(f"done coreset {method}: test error {100 * (1 - acc):.2f}%")

    active_fields = [
        "method", "budget_frac", "n_labeled", "test_acc", "test_error", "select_seconds", "target_seconds",
    ]
    active_path = os.path.join(args.out, "active.csv")
    if not args.skip_active:
        initial_n, b10, b20 = active_budgets(n)
        init_path = os.path.join(args.out, "active_initial.npy")
        if os.path.exists(init_path):
            initial = np.load(init_path)
        else:
            initial = random_subset(n, initial_n, args.seed + 7)
            np.save(init_path, initial)
        done = {row["method"] for row in read_csv(active_path)}
        log(f"active learning budgets: {len(initial)} -> {b10} -> {b20} (of {n})")

        def run_acquisition(method, selector_arch, seed_offset):
            labeled = [int(i) for i in initial]
            labeled_set = set(labeled)
            select_seconds = 0.0
            rounds = [b10 - len(labeled), b20 - b10]
            for r, k in enumerate(rounds):
                log(f"=== active {method} round {r + 1}: train {selector_arch} on {len(labeled)}, add {k} ===")
                result = fit(
                    selector_arch,
                    args.seed + seed_offset + r,
                    np.asarray(labeled, dtype=np.int64),
                    track=False,
                    snapshot=False,
                    uncertainty=True,
                    features=False,
                    eval_test=False,
                    prefix=f"[active {method} r{r + 1}]",
                    eval_every=0,
                    log_every=5,
                )
                select_seconds += float(result["seconds"])
                chosen = select_top_excluding(result["least_confidence"], labeled_set, k, args.seed + 300 + r)
                for idx in chosen:
                    labeled.append(int(idx))
                    labeled_set.add(int(idx))
            log(f"=== active {method}: train target on {len(labeled)} labeled ===")
            final = fit(
                "preact56",
                args.seed + seed_offset + 20,
                np.asarray(labeled, dtype=np.int64),
                track=False,
                snapshot=False,
                uncertainty=False,
                features=False,
                eval_test=True,
                prefix=f"[active {method} target]",
                eval_every=0,
                log_every=5,
            )
            acc = float(final["test_acc"])
            append_csv(active_path, {
                "method": method,
                "budget_frac": fmt(len(labeled) / n),
                "n_labeled": len(labeled),
                "test_acc": fmt(acc),
                "test_error": fmt(1.0 - acc),
                "select_seconds": fmt(select_seconds),
                "target_seconds": fmt(float(final["seconds"])),
            }, active_fields)
            log(f"done active {method}: test error {100 * (1 - acc):.2f}%  select {select_seconds:.0f}s")

        if "random" not in done:
            chosen = random_subset(n, b20, args.seed + 9)
            log(f"=== active random: train target on {len(chosen)} ===")
            final = fit(
                "preact56",
                args.seed + 410,
                chosen,
                track=False,
                snapshot=False,
                uncertainty=False,
                features=False,
                eval_test=True,
                prefix="[active random]",
                eval_every=0,
                log_every=5,
            )
            acc = float(final["test_acc"])
            append_csv(active_path, {
                "method": "random",
                "budget_frac": fmt(len(chosen) / n),
                "n_labeled": int(len(chosen)),
                "test_acc": fmt(acc),
                "test_error": fmt(1.0 - acc),
                "select_seconds": fmt(0.0),
                "target_seconds": fmt(float(final["seconds"])),
            }, active_fields)
            log(f"done active random: test error {100 * (1 - acc):.2f}%")
        else:
            log("resume active random")
        if "oracle_least_confidence" not in done:
            run_acquisition("oracle_least_confidence", "preact56", 200)
        else:
            log("resume active oracle")
        if "svp_least_confidence" not in done:
            run_acquisition("svp_least_confidence", "preact8", 250)
        else:
            log("resume active svp")

    coreset_rows = read_csv(coreset_path)
    active_rows = read_csv(active_path)
    by_core = {row["method"]: row for row in coreset_rows}
    by_active = {row["method"]: row for row in active_rows}
    e2e = {}
    if "full" in by_core and "forget_preact8" in by_core:
        full_s = float(by_core["full"]["train_seconds"])
        proxy_s = float(store["preact8_s0"]["meta"]["seconds"])
        subset_s = float(by_core["forget_preact8"]["train_seconds"])
        e2e = {
            "full_seconds": full_s,
            "proxy_seconds": proxy_s,
            "subset_seconds": subset_s,
            "svp_seconds": proxy_s + subset_s,
            "speedup_vs_training_on_all_data": full_s / (proxy_s + subset_s) if proxy_s + subset_s else None,
        }
    if "oracle_least_confidence" in by_active and "svp_least_confidence" in by_active:
        oracle_s = float(by_active["oracle_least_confidence"]["select_seconds"])
        svp_s = float(by_active["svp_least_confidence"]["select_seconds"])
        e2e["active_oracle_select_seconds"] = oracle_s
        e2e["active_svp_select_seconds"] = svp_s
        e2e["active_selection_speedup"] = (oracle_s / svp_s) if svp_s else None

    summary = {
        "env": {"torch": torch.__version__, "gpu": gpu, "cuda": device.type == "cuda"},
        "config": identity,
        "params": params,
        "ranking": {key: store[key]["meta"] for key in store},
        "correlations": corr_rows,
        "coreset": coreset_rows,
        "active": active_rows,
        "end_to_end": e2e,
        "paper_reference": {
            "fig3_forgetting_spearman_r20_vs_r164": 0.90,
            "table3_cifar10_least_confidence_error_at_20pct": {
                "random": 12.9,
                "resnet164": 10.4,
                "resnet20_proxy": 10.5,
            },
            "note": "Those numbers are ResNet-164, 181 epochs, 3-5 seeds. This run is not expected to match them absolutely.",
        },
    }
    dump_json(os.path.join(args.out, "summary.json"), summary)
    make_plots(args.out, args.snapshot_epoch, args.fraction)

    log("")
    log("=== Spearman (1 = same ranking) ===")
    for row in corr_rows:
        log(f"  {row['metric']:12s} {row['pair']:18s} {row['spearman']}")
    if coreset_rows:
        log("=== core-set, target test error ===")
        for row in coreset_rows:
            log(f"  {row['method']:24s} {100 * float(row['test_error']):6.2f}%   {float(row['train_seconds']):.0f}s")
    if active_rows:
        log("=== active learning, target test error ===")
        for row in active_rows:
            log(
                f"  {row['method']:28s} {100 * float(row['test_error']):6.2f}%   "
                f"select {float(row['select_seconds']):.0f}s  target {float(row['target_seconds']):.0f}s"
            )
    if e2e.get("speedup_vs_training_on_all_data"):
        log(
            f"end-to-end core-set (train R8 + train R56 on the R8 forgetting subset) "
            f"is {e2e['speedup_vs_training_on_all_data']:.2f}x vs training R56 on all the data"
        )
    if e2e.get("active_selection_speedup"):
        log(f"active-learning selection speed-up, R8 vs R56: {e2e['active_selection_speedup']:.2f}x")
    log("Paper reference, not a target for these absolute numbers:")
    log("  Fig. 3 forgetting Spearman, ResNet-20 vs ResNet-164 ≈ 0.90")
    log("  Fig. 2: a 50% forgetting core-set from a small proxy matches the oracle and beats random")
    log("  Table 3 CIFAR-10 least confidence @ 20%: random 12.9%, ResNet-164 10.4%, ResNet-20 proxy 10.5%")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
