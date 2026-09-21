import pandas as pd
from pathlib import Path

base_dir = Path("/home/noemi/atmoseer/data/ws/full")
output_dir = Path("/home/noemi/atmoseer/tcc-noemi-radar-nowcasting/maximos_m15_por_ano")

output_dir.mkdir(parents=True, exist_ok=True)

TOP_N = 20

for year in range(2024, 2025):
    all_rows = []

    for station_id in range(1, 84):
        path = base_dir / f"station_id={station_id}" / f"year={year}" / "data.parquet"

        if not path.exists():
            continue

        df = pd.read_parquet(path)

        if "m15" not in df.columns:
            continue

        df = df[["id", "nome", "observation_datetime", "m15"]].copy()
        df["station_id"] = station_id

        df["observation_datetime"] = pd.to_datetime(
            df["observation_datetime"],
            utc=True
        )

        df = df.dropna(subset=["m15"])

        all_rows.append(df)

    if not all_rows:
        print(f"{year}: sem dados")
        continue

    year_df = pd.concat(all_rows, ignore_index=True)

    top_df = (
        year_df
        .sort_values("m15", ascending=False)
        .head(TOP_N)[["station_id", "id", "nome", "observation_datetime", "m15"]]
    )

    output_path = output_dir / f"maximos_m15_{year}.csv"

    top_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("ANO:", year)
    print("Máximo direto dos parquets:", year_df["m15"].max())
    print("CSV salvo em:", output_path)
    print(top_df)