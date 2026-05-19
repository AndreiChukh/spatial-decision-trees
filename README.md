# Spatial Decision Trees

Реализация и сравнение четырёх алгоритмов:

* C4.5-DT и SIG-based Spatial Decision Tree (SDT)
* LTDT, FTSDT-fixed и FTSDT-adaptive

## Требования к окружению

* JDK 21 или новее (OpenJDK / Eclipse Temurin / Oracle).
* 8 ГБ оперативной памяти (на полной сцене Landsat 9 рекомендуется 16 ГБ).
* 2 ГБ свободного места на диске для распакованных данных и результатов.
* Сборка офлайн через Gradle wrapper; внешних сетевых зависимостей нет.

## Структура проекта

```
sdt_project/
├── build.gradle
├── settings.gradle
├── gradlew
├── gradlew.bat
├── data/                                            ← 7 файлов сцены Landsat 9
├── outputs_1/                                       ← результаты прогона C4.5-DT / SDT
├── outputs_2/                                       ← результаты прогона LTDT / FTSDT
├── README.md
└── src/
    └── main/
        └── java/
            └── sdt/
                ├── Main.java                        ← точка входа, парсинг CLI
                ├── check/
                │   ├── SatelliteSdtCheck.java      ← оркестратор C4.5-DT vs SDT
                │   └── Chapter5Check.java          ← оркестратор LTDT vs FTSDT
                ├── tree/
                │   ├── C45DecisionTree.java
                │   ├── SigSpatialDecisionTree.java
                │   └── InformationGainDecisionTree.java
                ├── ch5/
                │   ├── FocalSpatialDecisionTree.java
                │   ├── FocalNode.java
                │   └── FocalNeighborhoodMode.java
                ├── spatial/                         ← пространственный граф (CSR)
                ├── metrics/                         ← Entropy, IG, NSAR, Moran’s I, BB JC
                ├── tuning/                          ← AlphaTuner
                ├── split/                           ← SplitFinder
                ├── model/                           ← Sample, Node, SplitResult
                ├── io/                              ← TiffReader, SingleBandRaster
                └── viz/                             ← PNG-карты и графики
```

## Данные

Папка `data/` содержит 7 файлов Landsat 9:

```
data/
├── LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF       ← синий
├── LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF       ← зелёный
├── LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF       ← красный (для NDVI)
├── LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF       ← NIR (для NDVI)
├── LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF       ← SWIR-1
├── LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF       ← SWIR-2
└── LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF    ← маска качества
```

## Метки

Метки строятся из NDVI:

```
NDVI = (SR_B5 - SR_B4) / (SR_B5 + SR_B4)
high-NDVI = NDVI >= 0.10
low-NDVI  = NDVI <  0.10
```

Признаки деревьев: B2, B3, B6, B7. Каналы B4 и B5 в признаки не попадают, чтобы дерево не восстановило формулу NDVI напрямую.

---

## Запуск обоих экспериментов одной командой

Флаг `--all-trees` последовательно прогоняет оба сравнения и складывает результаты в `outputs_1/` и `outputs_2/` рядом друг с другом:

```bash
./gradlew run --args="data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF --all-trees"
```

При указании `--all-trees` остальные флаги (`--output-dir`, `--chapter5`) игнорируются.

---

## C4.5-DT vs SIG-based Spatial Decision Tree (SDT)

```bash
./gradlew run --args="data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF --output-dir outputs_1"
```

### Дополнительные параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--output-dir <path>` | `outputs_1` | Папка для результатов |
| `--ndvi-threshold <val>` | `0.10` | Порог NDVI для псевдо-метки |
| `--alpha <val>` | (auto) | Фиксированный alpha SDT; если не задан, ищется автоматически |
| `--alpha-end <val>` | `1.0` | Верхняя граница поиска alpha |
| `--min-node-size <n>` | `100` | Минимальный размер узла/листа дерева |
| `--train-validation-fraction <val>` | `0.70` | Доля верхних строк для train+validation |
| `--perf-limit <n>` | `5000` | Сколько samples используется для замера производительности графа |
| `--tree-print-limit <n>` | `64` | Макс. число листьев, при котором дерево печатается целиком |
| Позиц. 1: `psuSize` | `98` | Размер PSU в пикселях |
| Позиц. 2: `samplesPerPsu` | `1000` | Пикселей из каждого PSU |
| Позиц. 3: `trainPsuPerClass` | `8` | PSU для обучения на класс |
| Позиц. 4: `validationPsuPerClass` | `2` | PSU для валидации на класс |
| Позиц. 6: `testEvalStep` | `1` | Шаг (px) при оценке на тестовой зоне |
| Позиц. 9: `alphaStart` | `0.02` | Начало поиска alpha |
| Позиц. 10: `alphaStep` | `0.02` | Шаг поиска alpha |
| Позиц. 12: `neighborDistance` | `6.0` | Радиус пространственного соседства (px) |
| Позиц. 13: `randomSeed` | `42` | Зерно случайности |

---

## LTDT vs FTSDT-fixed vs FTSDT-adaptive

```bash
./gradlew run --args="data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF data/LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF --chapter5 --output-dir outputs_2"
```

### Дополнительные параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--output-dir <path>` | `outputs_2` | Папка для результатов |
| `--ndvi-threshold <val>` | `0.10` | Порог NDVI для псевдо-метки |
| `--min-node-size <n>` | `100` | Минимальный размер узла/листа |
| `--train-validation-fraction <val>` | `0.70` | Доля верхних строк для train |
| `--eval-step <n>` | `4` | Шаг (px) при сборке eval-выборки |
| `--max-eval-samples <n>` | `200000` | Макс. число eval-пикселей |
| `--seed <n>` | `42` | Зерно случайности |
| `--smax <n>` | `5` | Макс. радиус фокального окна (2s+1 × 2s+1) |

---

## Выходные файлы

### outputs_1

```
outputs_1/
├── table_c45_dt_confusion.csv          ← матрица ошибок C4.5-DT
├── table_sdt_confusion.csv             ← матрица ошибок SDT
├── table_bb_join_count.csv             ← BB Join Count статистика
├── alpha_tuning.csv                    ← кривая подбора alpha
├── alpha_tuning_chart.png
├── performance_table.csv
├── performance_chart.png
├── c45_sdt_error_chart.png             ← сравнение ошибок
├── bb_join_chart.png                   ← сравнение BB Join Count
├── fig_truth.png                       ← карта истинных меток
├── fig_c45_dt.png                      ← карта предсказаний C4.5-DT
├── fig_sdt.png                         ← карта предсказаний SDT
├── fig_combined.png                    ← три карты рядом
├── true_color_rgb_b4_b3_b2.png         ← RGB-визуализация
└── false_color_b5_b4_b3.png            ← ИК-визуализация
```

### outputs_2

```
outputs_2/
├── classification.csv                  ← метрики LTDT / FTSDT-fixed / FTSDT-adaptive
├── classification_chart.png
├── performance.csv
└── performance_chart.png
```
