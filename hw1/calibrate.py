"""калибровка параметров latency и energy

python calibrate.py --results results
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq


CONV_IDX = [
    i for i, l in enumerate(eq.LAYERS)
    if l.kind == "conv"
]

OTHER_IDX = [
    i for i, l in enumerate(eq.LAYERS)
    if l.kind != "conv"
]

# начальные значения параметров
X0 = dict(
    t0=5e-5,
    t_launch=1e-5,
    bw=250e9,
    rate=4e12,
    p=4.0,
)


def load(results_dir):
    df = pd.read_csv(os.path.join(results_dir, "measurements.csv"))

    with open(os.path.join(results_dir, "env.json")) as f:
        env = json.load(f)

    return df, env


def unpack(z, per_layer):
    # параметры оптимизируются в log scale
    t0, t_launch, bw, p = np.exp(z[:4])

    if per_layer:
        rate = np.empty(len(eq.LAYERS))

        # для каждой conv свой rate
        rate[CONV_IDX] = np.exp(
            z[4:4 + len(CONV_IDX)]
        )

        # для остальных слоев общий rate
        rate[OTHER_IDX] = np.exp(
            z[4 + len(CONV_IDX)]
        )

        rate = rate.tolist()

    else:
        rate = float(np.exp(z[4]))

    return dict(
        t0=float(t0),
        t_launch=float(t_launch),
        bw=float(bw),
        p=float(p),
        rate=rate,
    )


def fit_latency(s, b, t, per_layer, seed=0):
    n_rate = (
        len(CONV_IDX) + 1
        if per_layer
        else 1
    )

    base = np.log(
        [
            X0["t0"],
            X0["t_launch"],
            X0["bw"],
            X0["p"],
        ]
        + [X0["rate"]] * n_rate
    )

    lo = np.log(
        [1e-7, 1e-7, 1e9, 1.0]
        + [1e9] * n_rate
    )

    hi = np.log(
        [1e-1, 1e-2, 2e12, 50.0]
        + [5e13] * n_rate
    )

    # ошибка в log scale примерно соответствует относительной ошибке
    def resid(z):
        return (
            np.log(eq.latency(s, b, unpack(z, per_layer)))
            - np.log(t)
        )

    rng = np.random.default_rng(seed)

    best = None

    # несколько стартовых точек для нелинейной оптимизации
    for k in range(12):
        if k == 0:
            z0 = base
        else:
            z0 = np.clip(
                base + rng.normal(0, 1.0, base.size),
                lo + 1e-6,
                hi - 1e-6,
            )

        r = least_squares(
            resid,
            z0,
            bounds=(lo, hi),
            x_scale="jac",
        )

        if best is None or r.cost < best.cost:
            best = r

    return unpack(best.x, per_layer)


def fit_energy(s, b, e, theta_lat):
    t = eq.latency(s, b, theta_lat)

    # E = p_static*T + e_flop*FLOPs + e_byte*BytesMoved
    a = np.stack(
        [
            t,
            eq.flops(s, b),
            eq.bytes_moved(s, b),
        ],
        axis=1,
    )

    scale = a.max(axis=0)

    # относительная least-squares ошибка
    # коэффициенты ограничены значениями >= 0
    coef, _ = nnls(
        a / scale / e[:, None],
        np.ones_like(e),
    )

    coef = coef / scale

    return dict(
        p_static=float(coef[0]),
        e_flop=float(coef[1]),
        e_byte=float(coef[2]),
        latency=theta_lat,
    )


def ape_stats(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    mask = (
        np.isfinite(pred)
        & np.isfinite(true)
        & (true > 0)
    )

    pred = pred[mask]
    true = true[mask]

    if true.size == 0:
        return dict(
            mape=None,
            median_ape=None,
            max_ape=None,
            n=0,
        )

    ape = np.abs(pred - true) / true * 100

    return dict(
        mape=float(ape.mean()),
        median_ape=float(np.median(ape)),
        max_ape=float(ape.max()),
        n=int(ape.size),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")

    args = ap.parse_args()

    df, env = load(args.results)

    # probe точки не участвуют в calibration и validation
    ok = df[
        (df.status == "ok")
        & (df.is_probe == 0)
    ].copy()

    tr = ok[ok.is_validation == 0]
    va = ok[ok.is_validation == 1]

    S = lambda d: d.S.to_numpy(float)
    B = lambda d: d.B.to_numpy(float)

    metrics = {}
    thetas = {}

    # global использует один rate
    # per_layer использует отдельные rate для conv
    for name, per_layer in [
        ("global", False),
        ("per_layer", True),
    ]:
        th = fit_latency(
            S(tr),
            B(tr),
            tr.latency_s.to_numpy(),
            per_layer,
        )

        thetas[name] = th

        metrics[f"latency_{name}"] = {
            "train": ape_stats(
                eq.latency(S(tr), B(tr), th),
                tr.latency_s.to_numpy(),
            ),
            "validation": ape_stats(
                eq.latency(S(va), B(va), th),
                va.latency_s.to_numpy(),
            ),
        }

    # модель выбирается заранее
    # validation не используется для выбора
    selected = "global"

    theta_energy = None

    en = ok.dropna(subset=["energy_j"])
    en = en[en.energy_j > 0]

    en_tr = en[en.is_validation == 0]
    en_va = en[en.is_validation == 1]

    if len(en_tr) >= 3:
        theta_energy = fit_energy(
            S(en_tr),
            B(en_tr),
            en_tr.energy_j.to_numpy(),
            thetas[selected],
        )

        metrics["energy"] = {
            "train": ape_stats(
                eq.energy(S(en_tr), B(en_tr), theta_energy),
                en_tr.energy_j.to_numpy(),
            ),
            "validation": ape_stats(
                eq.energy(S(en_va), B(en_va), theta_energy),
                en_va.energy_j.to_numpy(),
            ),
        }

    # сравнение аналитической и измеренной памяти
    mem = ok

    metrics["memory"] = ape_stats(
        eq.memory(S(mem), B(mem)),
        mem.memory_peak_bytes.to_numpy(),
    )

    # проверка аналитических FLOPs через PyTorch
    fc = ok.dropna(subset=["flops_counted"])

    metrics["flops_conv_linear_vs_flopcounter"] = ape_stats(
        eq.flops(
            S(fc),
            B(fc),
            elementwise=False,
        ),
        fc.flops_counted.to_numpy(),
    )

    # проверка предсказания OOM
    # здесь probe точки тоже используются
    cap = env["free_memory_bytes_at_start"]

    pred_oom = eq.memory(S(df), B(df)) > cap
    real_oom = (df.status == "OOM").to_numpy()

    metrics["oom_prediction"] = dict(
        capacity_bytes=cap,
        true_pos=int((pred_oom & real_oom).sum()),
        false_pos=int((pred_oom & ~real_oom).sum()),
        false_neg=int((~pred_oom & real_oom).sum()),
        true_neg=int((~pred_oom & ~real_oom).sum()),
    )

    out = dict(
        latency=thetas[selected],
        latency_selected=selected,
        latency_global=thetas["global"],
        latency_per_layer=thetas["per_layer"],
        energy=theta_energy,
        idle_power_w=env.get("idle_power_w"),
        layer_names=eq.LAYER_NAMES,
        metrics=metrics,
    )

    with open(
        os.path.join(args.results, "theta.json"),
        "w",
    ) as f:
        json.dump(out, f, indent=2)

    # предсказания для всех измеренных конфигураций
    pred = df[
        [
            "S",
            "B",
            "status",
            "is_validation",
            "is_probe",
        ]
    ].copy()

    s = S(df)
    b = B(df)

    pred["flops"] = eq.flops(s, b)
    pred["bytes_moved"] = eq.bytes_moved(s, b)
    pred["memory_pred"] = eq.memory(s, b)

    pred["latency_pred"] = eq.latency(
        s,
        b,
        thetas[selected],
    )

    pred["regime_pred"] = np.array(
        ["launch", "memory", "compute"]
    )[
        eq.regime(
            s,
            b,
            thetas[selected],
        )
    ]

    if theta_energy is not None:
        pred["energy_pred"] = eq.energy(
            s,
            b,
            theta_energy,
        )

    pred.to_csv(
        os.path.join(
            args.results,
            "predictions.csv",
        ),
        index=False,
    )

    print(
        json.dumps(
            {
                k: v
                for k, v in out.items()
                if k != "layer_names"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
