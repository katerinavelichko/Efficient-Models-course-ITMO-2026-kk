from dataclasses import dataclass
import numpy as np


BYTES = 4  # FP32


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str # conv/relu/maxpool/avgpool/linear
    c_in: int
    c_out: int
    k: int = 1
    d_in: int = 1 # размер входа: S / d_in   d=0 -> H*W=1
    d_out: int = 1

    def hw_in(self, s):
        return 1.0 if self.d_in == 0 else (s / self.d_in) ** 2

    def hw_out(self, s):
        return 1.0 if self.d_out == 0 else (s / self.d_out) ** 2

    def elems_in(self, s):
        return self.c_in * self.hw_in(s)

    def elems_out(self, s):
        return self.c_out * self.hw_out(s)

    @property
    def params(self) -> int:
        if self.kind == "conv":
            return self.c_in * self.c_out * self.k ** 2
        if self.kind == "linear":
            return (self.c_in + 1) * self.c_out
        return 0

    @property
    def inplace(self) -> bool:
        return self.kind == "relu"

    def flops(self, s):
        if self.kind == "conv":
            return (
                2.0
                * self.k ** 2
                * self.c_in
                * self.c_out
                * self.hw_out(s)
            )

        if self.kind == "linear":
            return 2.0 * self.c_in * self.c_out + self.c_out

        if self.kind == "relu":
            return self.elems_out(s)

        if self.kind == "maxpool":
            return (self.k ** 2 - 1) * self.elems_out(s)

        if self.kind == "avgpool":
            return self.elems_in(s) + self.c_out

        raise ValueError(self.kind)

    def bytes_moved(self, s, b):
        # идеализированный объем переданных данных: вход + параметры + выход

        if self.inplace:
            return BYTES * 2.0 * b * self.elems_out(s)

        return BYTES * (
            b * self.elems_in(s)
            + self.params
            + b * self.elems_out(s)
        )


def _conv_relu(name, c_in, c_out, k, d_in, d_out):
    return [
        Layer(name, "conv", c_in, c_out, k, d_in, d_out),
        Layer(name + ".relu", "relu", c_out, c_out, 1, d_out, d_out),
    ]


LAYERS: list[Layer] = [
    *_conv_relu("conv1", 3, 32, 7, 1, 2), # S -> S/2
    Layer("maxpool", "maxpool", 32, 32, 3, 2, 4), # S/2 -> S/4
    *_conv_relu("conv2", 32, 64, 5, 4, 4), # S/4
    *_conv_relu("conv3", 64, 128, 3, 4, 8), # S/4 -> S/8
    *_conv_relu("conv4", 128, 256, 1, 8, 8), # S/8
    *_conv_relu("conv5", 256, 256, 3, 8, 16), # S/8 -> S/16
    *_conv_relu("conv6", 256, 512, 1, 16, 16), # S/16
    Layer("avgpool", "avgpool", 512, 512, 1, 16, 0),
    Layer("fc1", "linear", 512, 256, 1, 0, 0),
    Layer("fc1.relu", "relu", 256, 256, 1, 0, 0),
    Layer("fc2", "linear", 256, 100, 1, 0, 0),
]

LAYER_NAMES = [l.name for l in LAYERS]
N_PARAMS = sum(l.params for l in LAYERS)
INPUT_CHANNELS = 3


def _sb(image_size, batch):
    return np.broadcast_arrays(
        np.asarray(image_size, dtype=float),
        np.asarray(batch, dtype=float),
    )


def _out(x):
    return float(x) if np.ndim(x) == 0 else x


def flops(image_size, batch, elementwise: bool = True):
    # FLOPs одного forward pass
    # elementwise=False оставляет только Conv/Linear с 1 MAC=2 FLOPs


    s, b = _sb(image_size, batch)


    total = np.zeros_like(s)

    for l in LAYERS:
        if not elementwise and l.kind not in ("conv", "linear"):
    
            continue

        if not elementwise and l.kind == "linear":
            total += b * (2.0 * l.c_in * l.c_out)

            continue

        total += b * l.flops(s)

    return _out(total)


def flops_per_layer(image_size, batch):
    # FLOPs для каждого слоя
    s, b = _sb(image_size, batch)
    return {l.name: _out(b * l.flops(s)) for l in LAYERS}


# Memory
def activation_elems(image_size):
    # инпут + сумма всех активаций
    # inplace ReLU отдельную активацию не создает
    s = np.asarray(image_size, dtype=float)
    # входные каналы * размер изображения * размер изображения
    total = INPUT_CHANNELS * s ** 2

    for l in LAYERS:
        if not l.inplace:
            total += l.elems_out(s)
    return total


def memory(image_size, batch):
    # аналитическая модель памяти [байт]
    # параметры + инпут + сумма всех активаций

    s, b = _sb(image_size, batch)
    return _out(BYTES * (N_PARAMS + b * activation_elems(s)))


# байты переданных данных
def bytes_moved(image_size, batch):
    s, b = _sb(image_size, batch)
    total = np.zeros_like(s)

    for l in LAYERS:
        total += l.bytes_moved(s, b)

    return _out(total)


def bytes_moved_per_layer(image_size, batch):
    s, b = _sb(image_size, batch)
    return {l.name: _out(l.bytes_moved(s, b)) for l in LAYERS}



# Latency
# Для каждого слоя:
#  T_launch  = t_launch
#  T_compute = FLOPs / rate
#  T_memory  = BytesMoved / bw
#
# Latency = t0 + sum_l ||T_launch, T_compute, T_memory||_p
#
# p -> inf дает обычный max(), p = 1 дает сумму


def _rates(theta):
    r = np.asarray(theta["rate"], dtype=float)

    if r.ndim == 0:
        if r <= 0:
            raise ValueError("theta['rate'] должен быть положительным")

        return np.full(len(LAYERS), float(r))

    r = r.reshape(-1)

    if r.size != len(LAYERS):
        raise ValueError(
            f"theta['rate'] должен содержать {len(LAYERS)} значений, получено {r.size}"
        )

    if np.any(r <= 0):
        raise ValueError("Все значения rate должны быть положительными")

    return r


def _lp_max(terms, p):
    terms = np.stack(np.broadcast_arrays(*terms))

    if p < 1:
        raise ValueError("p должен быть >= 1")

    if np.isinf(p):
        return terms.max(axis=0)

    m = terms.max(axis=0)
    safe_m = np.where(m == 0, 1.0, m)
    out = (safe_m * ((terms / safe_m) ** p).sum(axis=0) ** (1.0 / p))

    return np.where(m == 0, 0.0, out)


def latency_terms(image_size, batch, theta) -> dict:
    # для каждого слоя возвращает (launch, compute, memory) время

    s, b = _sb(image_size, batch)
    rates = _rates(theta)

    out = {}

    for l, r in zip(LAYERS, rates):
        out[l.name] = (
            np.full_like(s, theta["t_launch"]),
            b * l.flops(s) / r,
            l.bytes_moved(s, b) / theta["bw"],
        )

    return out


def latency(image_size, batch, theta):
    # предсказанная latency одного forward pass [с]

    s, b = _sb(image_size, batch)

    p = float(theta.get("p", np.inf))

    total = np.full_like(s, theta["t0"])

    for launch, comp, mem in latency_terms(s, b, theta).values():
        total += _lp_max((launch, comp, mem), p)

    return _out(total)


def regime(image_size, batch, theta):
    # 0 = launch-bound
    # 1 = memory-bound
    # 2 = compute-bound

    s, b = _sb(image_size, batch)

    launch = np.full_like(s, theta["t0"])
    comp = np.zeros_like(s)
    mem = np.zeros_like(s)

    for t_l, t_c, t_m in latency_terms(s, b, theta).values():
        winner = np.argmax(
            np.stack(np.broadcast_arrays(t_l, t_m, t_c)),
            axis=0,
        )

        dom = np.maximum(np.maximum(t_l, t_c), t_m)

        launch += np.where(winner == 0, dom, 0)
        mem += np.where(winner == 1, dom, 0)
        comp += np.where(winner == 2, dom, 0)

    return _out(
        np.argmax(
            np.stack([launch, mem, comp]),
            axis=0,
        )
    )


# Energy = p_static * Latency + e_flop * FLOPs + e_byte * BytesMoved

def energy(image_size, batch, theta_energy):
    # предсказанная энергия одного forward pass [Дж]

    s, b = _sb(image_size, batch)

    t = latency(s, b, theta_energy["latency"])

    e = (
        theta_energy["p_static"] * t
        + theta_energy["e_flop"] * flops(s, b)
        + theta_energy["e_byte"] * bytes_moved(s, b)
    )

    return _out(e)
