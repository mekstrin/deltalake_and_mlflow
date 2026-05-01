# Лабораторная 3: Lakehouse на Polars + Delta Lake

## Быстрый старт

```bash
# 1. Скачать датасет с Kaggle и положить в data/raw/
#    https://www.kaggle.com/code/peymanradmanesh/flight-delay-analysis-2018-2024
#    Файл: flight_data_2018_2024.csv

# 2. Запустить весь пайплайн в Docker
docker-compose up --build

# 3. MLflow UI
open http://localhost:5000
```

## Структура проекта

```
lab_data/
├── data/
│   ├── raw/                  # flight_data_2018_2024.csv (Kaggle)
│   └── delta/
│       ├── bronze/flights/   # Delta: сырые данные
│       ├── silver/flights/   # Delta: очищенные данные (partitioned by year/month/day)
│       └── gold/
│           ├── analytics/    # Аналитические агрегаты
│           └── features/     # Feature table для ML
├── src/
│   ├── config.py
│   ├── bronze.py
│   ├── silver.py
│   ├── gold.py
│   ├── delta_utils.py
│   ├── ml.py
│   └── pipeline.py
├── logs/
├── mlflow/                   # MLflow artifacts
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Архитектура пайплайна

```
Raw CSV (Kaggle)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  BRONZE LAYER                                            │
│  Задача: принять сырые данные день за днём, сохранить     │
│          историю загрузок в версиях Delta                │
│  Формат: 31 append-батч → 31 версия таблицы              │
│                                                         │
│  Преобразования:                                         │
│   • schema_override для 20+ колонок (int/float/str)     │
│   • null_values=["", "NA", "N/A"] → null                │
│   • filter по дате (только day=X)                        │
│   • добавлены служебные поля:                           │
│       source_day  — дата среза                          │
│       source_month — месяц среза                        │
│       load_ts     — timestamp загрузки (UTC now)        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  SILVER LAYER                                            │
│  Задача: очистить данные, обогатить признаками,          │
│          убрать дубли при повторных запусках (MERGE)     │
│  Партиции: year/month/day                               │
│                                                         │
│  Преобразования:                                         │
│   1. Отфильтровать отменённые рейсы (Cancelled != 1)    │
│   2. Убрать строки с null в ключевых полях:              │
│      ArrDelay, DepTime, Origin, Dest, FlightDate         │
│   3. Удалить выбросы ArrDelay                              │
│      диапазон: [-60 мин, +600 мин] (10 часов)             │
│      (config: ARR_DELAY_MIN=-60, ARR_DELAY_MAX=600)        │
│   4. Нормализация текста (strip + UPPER):               │
│      IATA_Code_Operating_Airline, Origin, Dest           │
│   5. Парсинг FlightDate → FlightDate_parsed             │
│   6. Извлечение временных признаков:                    │
│      year, month, day, day_of_week, hour, season        │
│   7. Построение route = Origin + "-" + Dest             │
│   8. Ключ MERGE:                                        │
│      (FlightDate, IATA_Code_Operating_Airline,           │
│       Origin, Dest, DepTime, source_day)                │
│      — при повторном запуске обновляет, не дублирует     │
└─────────────────────────────────────────────────────────┘
        │
   ┌────┴───────────────────────────────────────────┐
   ▼                                                  ▼
┌────────────────────────┐              ┌─────────────────────────┐
│  GOLD/analytics         │              │  GOLD/features           │
│  Агрегаты по задержкам │              │  Feature table для ML    │
│                         │              │                         │
│  Группировка по:        │              │  Фичи (X):              │
│   • Origin              │              │   carrier, origin, dest  │
│   • IATA_Code_...       │              │   route, year/month/day │
│   • hour                │              │   day_of_week, hour      │
│   • season              │              │   season, distance      │
│   • year/month/day      │              │   DepDelay, TaxiOut      │
│                         │              │   CRSElapsedTime         │
│  Агрегации:             │              │                         │
│   avg_arr_delay         │              │  Целевые переменные (y): │
│   median_arr_delay      │              │   ArrDelay (регрессия)   │
│   std_arr_delay         │              │   is_delayed (бинарная,  │
│   flight_count          │              │    > 15 мин = 1)        │
│   pct_delayed (>15мин)  │              │                         │
└────────────────────────┘              └─────────────────────────┘
        │                                        │
        ▼                                        ▼
   Аналитические запросы              LightGBM / sklearn → MLflow
   (BI, дашборды)                    (RMSE, MAE, R², F1, ROC-AUC)
```

### Детализация по слоям

| Слой | Вход | Выход | Ключевые операции |
|------|------|-------|-------------------|
| **Bronze** | CSV (Kaggle) | Delta (append, 31 версия) | schema_override, null_values, фильтр по дате, служебные колонки |
| **Silver** | Bronze Delta | Delta (партиционированный) | filter cancelled/nulls/outliers, normalize text, derive temporal features (year/month/day/hour/season/route), MERGE upsert |
| **Gold/analytics** | Silver Delta | Delta (агрегаты) | group_by + mean/median/std/count по airport/carrier/hour/season |
| **Gold/features** | Silver Delta | Delta (фичи+таргеты) | select фичей, is_delayed бинаризация |

### Зачем нужен каждый слой

- **Bronze**: сырые данные сохранены как есть — можно откатиться к любой версии (time travel) или добавить новые дни без перезаписи
- **Silver**: deduplication (MERGE) + партиционирование = быстрые запросы по дате; очистка гарантирует качество аналитики
- **Gold/analytics**: pre-aggregated данные для BI — не нужно сканировать весь объём при каждом запросе
- **Gold/features**: ML-ready таблица — отфильтрована от null, содержит таргеты, готова для train/test split

## Polars `.explain()` — пример с пушдаунами

Запрос:
```python
import polars as pl

plan = (
    pl.scan_delta("data/delta/silver/flights")
    .filter(
        (pl.col("month") == 1) &
        (pl.col("day") == 15) &
        (pl.col("IATA_Code_Operating_Airline") == "DL") &
        (pl.col("ArrDelay").is_not_null())
    )
    .select(["Origin", "Dest", "ArrDelay", "hour", "Distance"])
    .group_by(["Origin", "Dest"])
    .agg([
        pl.col("ArrDelay").mean().alias("avg_delay"),
        pl.len().alias("cnt"),
    ])
    .explain(optimized=True)
)
print(plan)
```

Вывод:
```
AGGREGATE[maintain_order: false]
  [col("ArrDelay").mean().alias("avg_delay"), len().alias("cnt")] BY [col("Origin"), col("Dest")]
  FROM
  simple π 3/3 ["Origin", "Dest", "ArrDelay"]
    Parquet SCAN [/Users/m.ekstrin/PycharmProjects/lab_data/data/delta/silver/flights/year=2024/month=1/day=15/part-00000-7b3eccab-82dc-453c-a346-b13a8a60cba3-c000.zstd.parquet]
    PROJECT 6/130 COLUMNS
    SELECTION: [([([([(col("IATA_Code_Operating_Airline")) == ("DL")]) & (col("ArrDelay").is_not_null())]) & ([(col("day")) == (15)])]) & ([(col("month")) == (1)])]
```

Здесь:
- `Parquet SCAN [.../year=2024/month=1/day=15/...]` — читается только одна партиция (partition pruning)
- `PROJECT 6/130 COLUMNS` — загружены только 6 из 130 колонок (column pruning)
- `SELECTION` — predicate передан в Parquet reader (selection pushdown)

## Delta Lake возможности

| # | Функция | Где используется |
|---|---------|-----------------|
| 1 | **MERGE** | `silver.py` — upsert при повторном запуске | 
| 2 | **OPTIMIZE (compaction)** | `delta_utils.py` — объединяет мелкие Parquet файлы |
| 3 | **Z-ORDER** | `delta_utils.py` — кластеризация по `(OP_CARRIER, ORIGIN)` для быстрых фильтров |
| 4 | **Time travel** | `delta_utils.py` — `pl.scan_delta(..., version=0)` читает исходную загрузку |
| 5 | **Schema evolution** | `bronze.py` + `delta_utils.py` — `schema_mode="merge"` для добавления колонок |

## ML модели

| Задача | Модель | Метрика |
|--------|--------|---------|
| Регрессия (ARR_DELAY) | LightGBM Regressor | RMSE, MAE, R² |
| Регрессия (baseline) | LinearRegression | RMSE, MAE, R² |
| Классификация (is_delayed > 15 мин) | LightGBM Classifier | F1, ROC-AUC |
| Классификация (baseline) | LogisticRegression | F1, ROC-AUC |

MLflow логирует: параметры, метрики, модели, feature importance, версию gold Delta-таблицы.

## Датасет

Реальный датасет: [US Flight Delays 2018–2024 (Kaggle)](https://www.kaggle.com/code/peymanradmanesh/flight-delay-analysis-2018-2024)

Скачайте `flight_data_2018_2024.csv` и положите в `data/raw/`. Код загружает все дни января 2024 года.
