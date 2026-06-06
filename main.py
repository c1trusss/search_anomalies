from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PLOTS_DIR = Path("output/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = 'data_train'
OUTPUT_DIR = "output"

MIN_OBS = 20  # порог респондентов в группе
MIN_QUERIES = 3  # порог конкретных запросов конкретного респондента в конкретный день

ROBUST_Z_THRESHOLD = 3.5  # порог робуст скора


def load_data(data_dir):

    dfs = []

    for path in sorted(Path(data_dir).glob("month=*")):
        df = pd.read_parquet(path)

        month = path.name.replace("month=", "")
        df["Month"] = month

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def prepare_data(df):

    df = df.copy()

    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    df["BrandinDelivery"] = (
        pd.to_numeric(df["BrandinDelivery"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["researchdate"] = pd.to_datetime(df["researchdate"])
    df["Month"] = df["researchdate"].dt.to_period("M").astype(str)
    df["researchdate"] = df["researchdate"].dt.date

    mask = (
        (df["BrandinDelivery"] == 1)
        & (df["CategoryNameDelivery"].notna())
        & (df["CategoryNameDelivery"] != "")
    )

    return df.loc[mask].copy()


def build_daily_ots(df):

    grouped = (
        df.groupby(
            [
                "SubjectID",
                "researchdate",
                "CategoryNameDelivery",
                "BrandID",
                "Brand",
                "Month"
            ]
        )
        .agg(
            Weight=("Weight", "first"),
            query_count=("QueryText", "count")
        )
        .reset_index()
    )

    grouped["daily_ots"] = (
        grouped["Weight"]
        * grouped["query_count"]
    )

    return grouped


def detect_anomalies(ots):

    group_cols = [
        "CategoryNameDelivery",
        "BrandID",
        "researchdate"
    ]

    scored = ots.copy()

    scored["n_obs"] = (
        scored.groupby(group_cols)["daily_ots"]
        .transform("size")
    )

    scored["median"] = (
        scored.groupby(group_cols)["daily_ots"]
        .transform("median")
    )

    scored["mad"] = (
        scored.groupby(group_cols)["daily_ots"]
        .transform(lambda x: np.median(np.abs(x - np.median(x))))  # формула MAD
    )

    scored["robust_z"] = np.where(
        scored["mad"] == 0,
        0,
        0.6745 * (scored["daily_ots"] - scored["median"]) / scored["mad"]  # Формула Robust Z-score
    )

    scored["brand_total_ots"] = (
        scored.groupby(group_cols)["daily_ots"]
        .transform("sum")
    )

    anomaly_mask = (
        (scored["n_obs"] >= MIN_OBS) &

        (scored["query_count"] >= MIN_QUERIES) &

        (scored["robust_z"] > ROBUST_Z_THRESHOLD)
    )

    anomalies = scored.loc[anomaly_mask].copy()

    return anomalies


def build_reasons(anomalies):

    reasons = anomalies[
        [
            "SubjectID",
            "researchdate",
            "BrandID",
            "Brand",
            "CategoryNameDelivery",
            "daily_ots",
            "robust_z",
            "query_count",
            "Month"
        ]
    ].copy()

    reasons["score"] = reasons["robust_z"]

    reasons["threshold"] = ROBUST_Z_THRESHOLD

    reasons["reason"] = (

        "robust_z="
        + reasons["robust_z"].round(2).astype(str)

        + "%; queries="

        + reasons["query_count"]
        .astype(str)
    )

    return reasons


def build_anomalies(anomalies):

    return (
        anomalies[["SubjectID", "researchdate"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def save_outputs(anomalies_csv, reasons_csv):

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    anomalies_csv.to_csv(f"{OUTPUT_DIR}/anomalies.csv", index=False)
    reasons_csv.to_csv(f"{OUTPUT_DIR}/anomaly_reasons.csv", index=False)


def build_and_save_plots(all_data, ots, reasons):
    """
    Основная функция для построения трех видов графиков.
    """
    for month in reasons["Month"].unique():
        month_reasons = reasons[reasons["Month"] == month]
        month_all_data = all_data[all_data["Month"] == month]
        month_ots = ots[ots["Month"] == month]

        # Создаем папку для месяца
        month_plot_dir = PLOTS_DIR / month
        month_plot_dir.mkdir(exist_ok=True)

        print(f"Построение графиков для {month}...")

        # 1. Изменение суммарного OTS по категориям (Барчарт)
        try:
            plot_ots_change(month_all_data, month_ots, month_reasons, month, month_plot_dir)
        except Exception as e:
            print(f"  Ошибка построения графика OTS по категориям для {month}: {e}")

        # 2. Удаленные респонденты по датам (Сдвоенный барчарт)
        try:
            plot_removed_respondents(month_reasons, month_all_data, month, month_plot_dir)
        except Exception as e:
            print(f"  Ошибка построения графика удаленных респондентов для {month}: {e}")

        # 3. Изменение ежедневного OTS (Линейный график) — НАШ НОВЫЙ ГРАФИК
        try:
            plot_daily_ots_dynamics(month_all_data, month_ots, month_reasons, month, month_plot_dir)
        except Exception as e:
            print(f"  Ошибка построения линейного графика динамики OTS для {month}: {e}")


def plot_ots_change(all_data, ots_data, reasons_data, month, output_dir):
    """
    Построение графика изменения суммарного OTS по категориям.
    Сравнивает:
    - beforeFiltering (весь набор данных, BrandID != '')
    - betweenFilteringAndReweighing (OTS отфильтрованных данных за вычетом аномалий)
    """

    # 1. Расчет total_ots_before.
    # Фильтруем данные, чтобы оставить только записи с BrandID. Это будет база.
    before_mask = all_data["BrandID"].notna() & (all_data["BrandID"] != '')
    base_data = all_data[before_mask]
    total_ots_before = base_data.groupby("CategoryNameDelivery")["Weight"].sum()

    # 2. Расчет total_ots_between (после фильтрации и удаления аномалий, но до перевзвешивания).
    # Убираем аномалии из daily_ots
    removed_subject_ids = reasons_data["SubjectID"].unique()
    mask = ~ots_data["SubjectID"].isin(removed_subject_ids)
    kept_ots = ots_data[mask]
    total_ots_between = kept_ots.groupby("CategoryNameDelivery")["daily_ots"].sum()

    # 3. Объединение и расчет разницы
    # Приводим к общему списку категорий
    combined = pd.merge(total_ots_before, total_ots_between, left_index=True, right_index=True, how='outer').fillna(0)
    combined.columns = ["ots_before", "ots_between"]

    # Расчет разницы в процентах (ots_between - ots_before) / ots_before
    # Обработка деления на 0
    combined["diff_percent"] = np.where(
        combined["ots_before"] == 0,
        0,
        (combined["ots_between"] - combined["ots_before"]) / combined["ots_before"] * 100
    )
    combined["diff_percent"] = combined["diff_percent"].round(1)  # Округляем до 1 знака

    # Убираем категории с нулевой разницей
    plot_data = combined[combined["diff_percent"] != 0].sort_values("diff_percent")

    if plot_data.empty:
        print(f"    Нет данных для графика OTS за {month}")
        return

    # Построение графика
    fig, ax = plt.subplots(figsize=(16, 10))
    bars = ax.bar(plot_data.index, plot_data["diff_percent"], color='tab:blue')

    # Настройка осей и подписей
    ax.set_title(
        f"Month={month}. Гистограмма изменения суммарного OTS по категориям в % (beforeFiltering - betweenFilteringAndReweighing)")
    ax.set_ylabel("%")
    ax.set_xlabel("Категория")

    # Добавляем подписи значений
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}", ha='center', va='top', fontsize=9)

    # Оформление
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Сохранение
    plt.savefig(output_dir / "ots_change.png")
    plt.close()


def plot_removed_respondents(reasons_data, all_data, month, output_dir):
    """
    График количества удаленных респондентов по датам.
    """
    if reasons_data.empty:
        print(f"    Нет удаленных респондентов для графика за {month}")
        return

    # Уникальные пары [SubjectID, researchdate]
    removed_pairs = reasons_data[["SubjectID", "researchdate"]].drop_duplicates()

    # Считаем количество удаленных респондентов в день и СРАЗУ даем имя колонке
    removed_counts = (
        removed_pairs.groupby("researchdate")
        .size()
        .reset_index(name="TotalRemoved")
    )

    # Считаем уникальных респондентов в день и тоже даем имя
    unique_removed_counts = (
        removed_pairs.groupby("researchdate")["SubjectID"]
        .nunique()
        .reset_index(name="UniqueRemoved")
    )

    # Теперь мерджим по колонке 'researchdate', так как это теперь полноценные DataFrame
    plot_data = pd.merge(removed_counts, unique_removed_counts, on="researchdate", how="outer").fillna(0)

    # Сортируем по дате, чтобы на графике всё шло по порядку
    plot_data = plot_data.sort_values("researchdate")

    # Расчет общего числа удаленных для заголовка
    total_removed = reasons_data["SubjectID"].nunique()
    total_reasons = len(reasons_data)

    # Построение графика
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(plot_data))
    bar_width = 0.35  # чуть-чуть сузим, чтобы смотрелось аккуратнее

    # Строим бары
    ax.bar(x - bar_width / 2, plot_data["TotalRemoved"], width=bar_width, label="Удалено (всего)", color='tab:blue')
    ax.bar(x + bar_width / 2, plot_data["UniqueRemoved"], width=bar_width, label="Удалено (уникальных)",
           color='darkorange')

    # Настройка осей и заголовка
    ax.set_title(
        f"Month={month}. Гистограмма количества удаленных респондентов. Всего удалено {total_reasons} респондентов, из них {total_removed} уникальных")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Количество outlier'ов")

    # Превращаем даты '2025-05-01' -> в красивые '01', '02' и т.д.
    labels = plot_data["researchdate"].astype(str).str.split('-').str[-1]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)  # применил 0 вместо 90, так как "01", "02" короткие и влезут горизонтально

    # Оформление
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Сохранение
    plt.savefig(output_dir / "removed_respondents.png")
    plt.close()


def plot_daily_ots_dynamics(all_data, ots_data, reasons_data, month, output_dir):
    """
    Построение линейного графика изменения ежедневного OTS (в тыс.)
    Сравнивает OTS до фильтрации и после удаления аномалий по дням.
    """
    # 1. Считаем OTS до фильтрации (по исходным данным, где есть BrandID)
    before_mask = all_data["BrandID"].notna() & (all_data["BrandID"] != '')
    base_data = all_data[before_mask].copy()
    # Так как в исходных данных OTS — это Weight * количество строк (если нет query_count),
    # либо просто используем Weight как базовый OTS. Для точности считаем сумму Weight.
    daily_before = base_data.groupby("researchdate")["Weight"].sum() / 1000.0  # переводим в тыс.

    # 2. Считаем OTS после фильтрации и удаления аномалий
    removed_subject_ids = reasons_data["SubjectID"].unique()
    mask = ~ots_data["SubjectID"].isin(removed_subject_ids)
    kept_ots = ots_data[mask]
    daily_after = kept_ots.groupby("researchdate")["daily_ots"].sum() / 1000.0  # переводим в тыс.

    # 3. Объединяем данные по датам
    plot_data = pd.merge(daily_before, daily_after, left_index=True, right_index=True, how='outer').fillna(0)
    plot_data.columns = ["ots_before", "ots_after"]
    plot_data = plot_data.sort_values("researchdate")

    if plot_data.empty:
        print(f"    Нет данных для графика динамики OTS за {month}")
        return

    # Метрики для подзаголовка
    avg_before = plot_data["ots_before"].mean()
    avg_after = plot_data["ots_after"].mean()
    pct_kept = (avg_after / avg_before * 100) if avg_before != 0 else 0

    # Построение графика
    fig, ax = plt.subplots(figsize=(16, 8))

    # Строим линии
    ax.plot(plot_data.index.astype(str), plot_data["ots_before"], color='red', label='OTS_beforeFilter')
    ax.plot(plot_data.index.astype(str), plot_data["ots_after"], color='green', label='OTS_betweenFilterAndReweighing')

    # Настройки заголовок и подзаголовок в стиле примера
    title_text = f"Month={month}. Изменение ежедневного OTS (beforeFiltering - betweenFilteringAndReweighing)"
    subtitle_text = f"'avg_ots_before' = {avg_before:.2f}, 'avg_ots_after' = {avg_after:.2f}, % = {pct_kept:.2f}"
    ax.set_title(f"{title_text}\n{subtitle_text}")

    ax.set_xlabel("Дата")
    ax.set_ylabel("OTS (в тыс.)")

    # Форматируем ось X (оставляем только дни "01", "02"...)
    labels = plot_data.index.astype(str).str.split('-').str[-1]
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(labels)

    # Динамическая сетка по Y как на скрине
    max_val = max(plot_data["ots_before"].max(), plot_data["ots_after"].max())
    ax.set_ylim(0, max_val * 1.1)
    ax.set_yticks(np.linspace(0, max_val * 1.1, 23))  # Создает частые деления сетки

    # Оформление
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()

    # Сохранение
    plt.savefig(output_dir / "daily_ots_dynamics.png")
    plt.close()


def main():
    print("Загрузка данных...")
    raw = load_data(DATA_DIR)

    print("Подготовка данных...")
    filtered = prepare_data(raw)

    print("Расчет daily_ots...")
    ots = build_daily_ots(filtered)

    print("Поиск аномалий...")
    anomalies = detect_anomalies(ots)

    reasons_csv = build_reasons(anomalies)
    anomalies_csv = build_anomalies(anomalies)

    save_outputs(
        anomalies_csv,
        reasons_csv
    )

    print()
    print(f"Найдено аномалий: {len(reasons_csv)}")
    print(f"Удалено пар: {len(anomalies_csv)}")

    # -- НОВАЯ ЧАСТЬ: ПОСТРОЕНИЕ ГРАФИКОВ --
    print("\n--- НАЧИНАЕМ ПОСТРОЕНИЕ ГРАФИКОВ ---")

    all_filtered = prepare_data(raw)

    build_and_save_plots(all_filtered, ots, reasons_csv)

    print("\nГотово! Графики сохранены в output/plots")


if __name__ == '__main__':
    main()
