import argparse
import csv
import math
import statistics
from pathlib import Path

MISSING_VALUES = {"", "na", "n/a", "null", "none", "-"}
NON_NUMERIC_COLUMNS = {"class_id", "class_name"}


def read_csv_rows(input_path: Path):
    """Read the CSV and repair rows with a single missing/extra blank cell."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, skipinitialspace=True)

        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            raise ValueError("File CSV kosong.")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            row = [cell.strip() for cell in row]

            if not row or all(cell == "" for cell in row):
                continue

            expected = len(header)

            # Jika satu kolom kurang, anggap kolom yang terlewat adalah
            # kolom kosong tepat sebelum nilai terakhir (annotation_count).
            if len(row) == expected - 1:
                row.insert(len(row) - 1, "")

            # Jika ada kolom lebih, hapus hanya sel kosong tambahan.
            # Nilai terakhir dipertahankan karena merupakan annotation_count.
            elif len(row) > expected:
                extra_cells = len(row) - expected
                for _ in range(extra_cells):
                    blank_positions = [
                        index for index, value in enumerate(row[:-1])
                        if value == ""
                    ]
                    if not blank_positions:
                        raise ValueError(
                            f"Baris {line_number} memiliki kolom berlebih "
                            "yang bukan kolom kosong. Periksa data sumber."
                        )
                    row.pop(blank_positions[-1])

            if len(row) != expected:
                raise ValueError(
                    f"Baris {line_number}: ditemukan {len(row)} kolom, "
                    f"seharusnya {expected}. Periksa data sumber."
                )

            rows.append(dict(zip(header, row)))

    return header, rows


def to_number(value, column_name, row_number):
    """Convert a CSV cell to float; represent missing values as None."""
    cleaned = value.strip()
    if cleaned.lower() in MISSING_VALUES:
        return None

    # Mendukung desimal bertitik maupun koma.
    cleaned = cleaned.replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError as error:
        raise ValueError(
            f"Nilai '{value}' pada kolom '{column_name}', "
            f"baris data {row_number} bukan angka yang valid."
        ) from error

    if not math.isfinite(number):
        return None
    return number


def percentile(sorted_values, percent):
    """Calculate a percentile using linear interpolation."""
    if not sorted_values:
        return None

    position = (len(sorted_values) - 1) * percent
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def calculate_statistics(header, rows):
    """Calculate descriptive statistics for every numeric column."""
    numeric_columns = [
        column for column in header
        if column.lower() not in NON_NUMERIC_COLUMNS
    ]

    results = []
    for column in numeric_columns:
        values = [
            to_number(row[column], column, row_index)
            for row_index, row in enumerate(rows, start=1)
        ]
        valid_values = sorted(value for value in values if value is not None)
        count = len(valid_values)
        missing_count = len(values) - count

        if count:
            mean_value = statistics.mean(valid_values)
            std_value = statistics.stdev(valid_values) if count > 1 else None
            min_value = valid_values[0]
            p25 = percentile(valid_values, 0.25)
            median = percentile(valid_values, 0.50)
            p75 = percentile(valid_values, 0.75)
            max_value = valid_values[-1]
        else:
            mean_value = std_value = min_value = None
            p25 = median = p75 = max_value = None

        results.append({
            "feature": column,
            "count": count,
            "missing_count": missing_count,
            "mean": mean_value,
            "std": std_value,
            "min": min_value,
            "25%": p25,
            "50%": median,
            "75%": p75,
            "max": max_value,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Membuat statistics.csv dari classes_nutritions.csv."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="docs/classes_nutritions.csv",
        help="Path file CSV input (default: classes_nutritions.csv).",
    )
    parser.add_argument(
        "--output",
        default="statistics.csv",
        help="Path file CSV output (default: statistics.csv).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    header, rows = read_csv_rows(input_path)
    stats_rows = calculate_statistics(header, rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "feature", "count", "missing_count", "mean", "std",
        "min", "25%", "50%", "75%", "max",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"Berhasil membuat: {output_path}")
    print(f"Jumlah kelas/baris data: {len(rows)}")
    print(f"Jumlah fitur numerik: {len(stats_rows)}")


if __name__ == "__main__":
    main()
