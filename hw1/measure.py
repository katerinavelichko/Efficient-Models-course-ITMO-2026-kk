import argparse
import csv
import json
import os
import platform
import threading
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from models import build_model


BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

N_RAND_S = 4
N_RAND_B = 3

# дополнительные точки для поиска границы OOM
# в calibration и validation они не участвуют
PROBE_S = [384, 512]
PROBE_B = [384, 512, 768, 1024, 1280, 1536, 2048]

FIELDS = [
    "S",
    "B",
    "status",
    "latency_s",
    "latency_p25_s",
    "latency_p75_s",
    "n_reps",
    "memory_peak_bytes",
    "memory_static_bytes",
    "energy_j",
    "power_w",
    "energy_window_s",
    "sm_clock_mhz",
    "temp_c",
    "flops_counted",
    "is_validation",
    "is_probe",
]


def set_flags():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_grid(seed=0):
    rng = np.random.default_rng(seed)

    # случайные S кратны 16 и не входят в базовую сетку
    s_pool = [
        s for s in range(32, 513, 16)
        if s not in BASE_S
    ]

    # случайные B не являются степенями двойки
    b_pool = [
        b for b in range(1, 257)
        if b & (b - 1) != 0
    ]

    rand_s = sorted(
        int(x)
        for x in rng.choice(s_pool, N_RAND_S, replace=False)
    )

    rand_b = sorted(
        int(x)
        for x in rng.choice(b_pool, N_RAND_B, replace=False)
    )

    configs = []

    for s in sorted(BASE_S + rand_s):
        for b in sorted(BASE_B + rand_b):
            configs.append(
                dict(
                    S=s,
                    B=b,
                    is_validation=(s in rand_s or b in rand_b),
                    is_probe=False,
                )
            )

    for s in PROBE_S:
        for b in PROBE_B:
            configs.append(
                dict(
                    S=s,
                    B=b,
                    is_validation=False,
                    is_probe=True,
                )
            )

    return configs, rand_s, rand_b


class GpuMonitor:

    # сначала используем аппаратный счетчик энергии NVML
    # а если он недоступен интегрируем измерения мощности

    def __init__(self, device_index=0):
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()

        self.h = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.has_counter = True
        except pynvml.NVMLError:
            self.has_counter = False

        self._samples = []
        self._stop = threading.Event()

    def power_w(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0

    def sm_clock(self):
        return float(
            self.nv.nvmlDeviceGetClockInfo(
                self.h,
                self.nv.NVML_CLOCK_SM,
            )
        )

    def temperature(self):
        return float(
            self.nv.nvmlDeviceGetTemperature(
                self.h,
                self.nv.NVML_TEMPERATURE_GPU,
            )
        )

    def _sampler(self):
        while not self._stop.is_set():
            self._samples.append(
                (time.perf_counter(), self.power_w())
            )
            time.sleep(0.005)

    def start(self):
        self._t0 = time.perf_counter()

        if self.has_counter:
            self._e0 = self.nv.nvmlDeviceGetTotalEnergyConsumption(
                self.h
            )
        else:
            self._samples = []
            self._stop = threading.Event()

            self._thread = threading.Thread(
                target=self._sampler,
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        # возвращает энергию в джоулях и длительность окна в секундах
        dt = time.perf_counter() - self._t0

        if self.has_counter:
            e = (
                self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)
                - self._e0
            ) / 1000.0

        else:
            self._stop.set()
            self._thread.join()

            if len(self._samples) < 2:
                raise RuntimeError(
                    "недостаточно измерений мощности NVML"
                )

            t, p = np.array(self._samples).T

            trapz = getattr(np, "trapezoid", None) or np.trapz
            e = float(trapz(p, t))

        return e, dt


@torch.inference_mode()
def measure_one(
    model,
    s,
    b,
    mon,
    min_reps=10,
    max_reps=300,
    latency_budget_s=0.5,
    energy_window_s=2.0,
    probe=False,
):

    row = dict(S=s, B=b)

    x = None
    y = None

    try:
        x = torch.randn(
            b,
            3,
            s,
            s,
            device="cuda",
        )

        # проверочный FLOP count PyTorch
        with FlopCounterMode(display=False) as fc:
            y = model(x)

        row["flops_counted"] = fc.get_total_flops()

        # прогрев
        for _ in range(2):
            y = model(x)

        torch.cuda.synchronize()

        # peak memory одного forward pass
        y = None
        torch.cuda.reset_peak_memory_stats()

        row["memory_static_bytes"] = (torch.cuda.memory_allocated())
        y = model(x)
        torch.cuda.synchronize()

        row["memory_peak_bytes"] = (torch.cuda.max_memory_allocated())

        # один запуск для выбора числа повторений
        y = None

        torch.cuda.synchronize()
        t = time.perf_counter()

        y = model(x)

        torch.cuda.synchronize()
        one = time.perf_counter() - t

        y = None

        n = (
            1
            if probe
            else int(
                np.clip(
                    latency_budget_s / max(one, 1e-6),
                    min_reps,
                    max_reps,
                )
            )
        )

        # latency как медиана нескольких запусков
        times = []

        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()

            y = model(x)

            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)

            y = None

        row.update(
            latency_s=float(np.median(times)),
            latency_p25_s=float(np.percentile(times, 25)),
            latency_p75_s=float(np.percentile(times, 75)),
            n_reps=n,
        )

        # энергия измеряется на серии запусков для уменьшения шума
        if mon is not None and not probe:
            n_e = max(
                3,
                int(
                    np.ceil(
                        energy_window_s / row["latency_s"]
                    )
                ),
            )

            torch.cuda.synchronize()
            mon.start()

            for _ in range(n_e):
                y = model(x)

            torch.cuda.synchronize()

            e, dt = mon.stop()

            row.update(
                energy_j=e / n_e,
                power_w=e / dt,
                energy_window_s=dt,
                sm_clock_mhz=mon.sm_clock(),
                temp_c=mon.temperature(),
            )

        row["status"] = "ok"

    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"

    finally:
        del x, y
        torch.cuda.empty_cache()

    return row


def env_info():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()

    info = dict(
        gpu=p.name,
        sm_count=p.multi_processor_count,
        compute_capability=f"{p.major}.{p.minor}",
        total_memory_bytes=total,
        free_memory_bytes_at_start=free,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        python=platform.python_version(),
    )

    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        info["driver"] = pynvml.nvmlSystemGetDriverVersion()

        info["power_limit_w"] = (
            pynvml.nvmlDeviceGetPowerManagementLimit(h)
            / 1000.0
        )

        info["max_sm_clock_mhz"] = (
            pynvml.nvmlDeviceGetMaxClockInfo(
                h,
                pynvml.NVML_CLOCK_SM,
            )
        )

    except Exception as exc: # ошибка NVML не должна останавливать измерения
        info["nvml_error"] = repr(exc)

    return info


def idle_power(mon, seconds=3.0):
    # базовая мощность GPU без выполнения модели
    torch.cuda.synchronize()
    time.sleep(1.0)

    mon.start()
    time.sleep(seconds)

    e, dt = mon.stop()

    return e / dt


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    set_flags()

    os.makedirs(args.out, exist_ok=True)

    csv_path = os.path.join(
        args.out,
        "measurements.csv",
    )

    configs, rand_s, rand_b = make_grid(args.seed)

    if args.quick:
        configs = [
            c
            for c in configs
            if c["S"] in (32, 224)
            and c["B"] in (1, 16)
        ]

    try:
        mon = GpuMonitor(0)

    except Exception as exc:
        print(
            "NVML недоступен, энергия измеряться не будет:",
            exc,
        )
        mon = None

    info = env_info()

    info.update(
        rand_s=rand_s,
        rand_b=rand_b,
        seed=args.seed,
        energy_source=(
            None
            if mon is None
            else (
                "nvml_counter"
                if mon.has_counter
                else "power_sampling"
            )
        ),
    )

    if mon is not None:
        info["idle_power_w"] = idle_power(mon)

    with open(
        os.path.join(args.out, "env.json"),
        "w",
    ) as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))

    # уже измеренные конфигурации пропускаем
    done = set()

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {
                (int(r["S"]), int(r["B"]))
                for r in csv.DictReader(f)
            }

    todo = [
        c
        for c in configs
        if (c["S"], c["B"]) not in done
    ]

    # случайный порядок уменьшает влияние нагрева и изменения частот
    # probe точки запускаются последними
    rng = np.random.default_rng(args.seed)

    grid = [
        c
        for c in todo
        if not c["is_probe"]
    ]

    grid = [
        grid[i]
        for i in rng.permutation(len(grid))
    ]

    todo = (
        grid
        + [
            c
            for c in todo
            if c["is_probe"]
        ]
    )

    model = build_model("cuda")

    new_file = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        if new_file:
            w.writeheader()

        for i, c in enumerate(todo):
            t = time.time()

            row = measure_one(
                model,
                c["S"],
                c["B"],
                mon,
                probe=c["is_probe"],
            )

            row.update(
                is_validation=int(c["is_validation"]),
                is_probe=int(c["is_probe"]),
            )

            w.writerow(row)
            f.flush()

            lat = row.get("latency_s")

            print(
                f"[{i + 1}/{len(todo)}] "
                f"S={c['S']:4d} "
                f"B={c['B']:5d} "
                f"{row['status']:3s} "
                f"lat={'-' if lat is None else f'{lat * 1e3:9.3f} ms'} "
                f"mem={row.get('memory_peak_bytes', 0) / 2**20:9.1f} MiB "
                f"E={row.get('energy_j', float('nan')):.4f} J "
                f"({time.time() - t:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
