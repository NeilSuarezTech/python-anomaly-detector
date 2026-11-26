"""Detect anomalies in tabular data with Isolation Forest.

Reads a CSV, scores every row on the chosen numeric columns, and writes:
  - <out>/anomalies.csv   all rows with an anomaly flag and score
  - <out>/anomalies.xlsx  same data, anomalies highlighted, plus a summary sheet
  - <out>/anomalies_3d.png  3D scatter of the first three columns

Usage:
  python anomaly_detector.py sample_data/sensor_readings.csv
  python anomaly_detector.py data.csv --columns amount,hour,items --contamination 0.02
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display (servers, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def detect(df: pd.DataFrame, columns: list[str], contamination: float, seed: int = 42) -> pd.DataFrame:
    """Return a copy of df with 'anomaly_score' (higher = more unusual) and 'is_anomaly'."""
    data = df[columns].dropna()
    X = StandardScaler().fit_transform(data)
    model = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    result = df.copy()
    result.loc[data.index, "anomaly_score"] = (-model.score_samples(X)).round(4)
    result.loc[data.index, "is_anomaly"] = model.predict(X) == -1
    result["is_anomaly"] = result["is_anomaly"].fillna(False).astype(bool)
    return result


def plot_3d(result: pd.DataFrame, columns: list[str], path: Path) -> None:
    x, y, z = columns[:3]
    normal, anomalies = result[~result["is_anomaly"]], result[result["is_anomaly"]]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(normal[x], normal[y], normal[z], c="#3b82f6", s=14, alpha=0.5, label=f"Normal ({len(normal)})")
    ax.scatter(anomalies[x], anomalies[y], anomalies[z], c="#ef4444", s=60, edgecolors="black",
               label=f"Anomaly ({len(anomalies)})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title("Isolation Forest anomaly detection")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_excel(result: pd.DataFrame, columns: list[str], path: Path) -> None:
    summary = pd.DataFrame({
        "metric": ["rows analysed", "anomalies found", "anomaly rate"],
        "value": [len(result), int(result["is_anomaly"].sum()), f"{result['is_anomaly'].mean():.1%}"],
    })
    stats = result.groupby("is_anomaly")[columns].mean().round(2)
    stats.index = stats.index.map({False: "normal (mean)", True: "anomaly (mean)"})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        result.sort_values("anomaly_score", ascending=False).to_excel(xw, sheet_name="Data", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        stats.to_excel(xw, sheet_name="Summary", startrow=len(summary) + 2)
        from openpyxl.styles import PatternFill
        ws = xw.sheets["Data"]
        red = PatternFill("solid", fgColor="FDE2E2")
        flag_col = list(result.columns).index("is_anomaly") + 1
        for row in ws.iter_rows(min_row=2):
            if row[flag_col - 1].value:
                for cell in row:
                    cell.fill = red
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(12, len(str(col[0].value)) + 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="sample_data/sensor_readings.csv", help="input CSV file")
    parser.add_argument("--columns", help="comma-separated numeric columns (default: all numeric)")
    parser.add_argument("--contamination", type=float, default=0.03, help="expected share of anomalies (default 0.03)")
    parser.add_argument("--out", default="output", help="output folder (default: output)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    columns = args.columns.split(",") if args.columns else df.select_dtypes("number").columns.tolist()
    if not columns:
        parser.error("no numeric columns found; use --columns")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = detect(df, columns, args.contamination)
    result.to_csv(out / "anomalies.csv", index=False)
    write_excel(result, columns, out / "anomalies.xlsx")
    if len(columns) >= 3:
        plot_3d(result, columns, out / "anomalies_3d.png")

    found = result[result["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    print(f"{len(found)} anomalies in {len(result)} rows (columns: {', '.join(columns)})")
    print(found.head(10).to_string(index=False))
    print(f"Results saved in {out}/")


if __name__ == "__main__":
    main()
