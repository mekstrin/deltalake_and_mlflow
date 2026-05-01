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
Raw CSV
      │
      ▼  append per day
  BRONZE (Delta)          ← schema_mode=merge (schema evolution)
      │
      ▼  MERGE by flight key
  SILVER (Delta)          ← partition_by=["year","month","day"]
      │
      ├──▶ GOLD/analytics  ← агрегаты по airport/carrier/hour/season
      │
      └──▶ GOLD/features   ← ML feature table (ARR_DELAY, is_delayed)
                │
                ▼
            LightGBM / sklearn   ──▶  MLflow
```

## Bronze Layer

Каждый день января 2024 года загружается отдельным батчем в `mode="append"`, что создаёт
31 версию Delta (версия 0 = 1 января, версия 30 = 31 января).
Имитирует реальный инкрементальный приём данных (день за днём).

```python
for day in DAYS:
    df = load_day(day)
    write_deltalake(BRONZE_PATH, df, mode="append", schema_mode="merge")
```

## Silver Layer: MERGE

При повторном запуске пайплайна данные **обновляются**, а не дублируются.
Ключ MERGE: `(FlightDate, IATA_Code_Operating_Airline, Origin, Dest, DepTime, source_day)`.

```python
dt.merge(source=df, predicate=predicate, source_alias="s", target_alias="t")
  .when_matched_update_all()
  .when_not_matched_insert_all()
  .execute()
```

## Партиционирование Silver

Партиции `["year", "month", "day"]` выбраны по следующим соображениям:

- **Типичные запросы** фильтруют по временному диапазону (например, задержки за день, или за несколько дней).
  Polars/DeltaLake пропускают нерелевантные партиции целиком.
- **Датасет содержит данные только за январь 2024** — при менее гранулярном выборе партициирования продемонстировать отсечение по партициям не вышло бы.

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
