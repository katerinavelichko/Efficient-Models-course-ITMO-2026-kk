# ДЗ-2. Статья Selection via Proxy

Пересказ статьи в презентации [hw2/Efficient_NN_HW2.pdf](./Efficient_NN_HW2.pdf).

Статья: [Coleman et al., "Selection via Proxy: Efficient Data Selection for Deep Learning", ICLR 2020](https://arxiv.org/abs/1906.11829). Код авторов: https://github.com/stanford-futuredata/selection-via-proxy

Дешевая proxy (меньше слоев или меньше эпох) ранжирует примеры почти так же, как целевая сеть. Отбор данных считают по proxy, а target учат уже на выбранном подмножестве. Здесь это уменьшенный прогон на CIFAR-10, один сид: proxy PreAct-ResNet-8, промежуточная PreAct-ResNet-20, target PreAct-ResNet-56, 30 эпох. Считаются корреляции рангов, core-set 50% и active learning до 20% меток. Результаты в [REPORT.md](./REPORT.md).


## Структура

```
hw2/
├── Efficient_NN_HW2.pdf   # пересказ статьи
├── README.md
├── REPORT.md              # что воспроизвелось
├── models.py              # PreAct-ResNet-8/20/56
├── data.py                # CIFAR-10, аугментация как у авторов
├── selection.py           # forgetting, entropy, least confidence, k-centers
├── train_loop.py          # SGD и подсчет forgetting
├── experiment.py          # корреляции, core-set, active learning
└── results/
    ├── figures/           # графики из отчета
    ├── *.csv
    └── summary.json
```
