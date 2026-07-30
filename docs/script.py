from pathlib import Path
import shutil

# ===== CONFIG =====
SOURCE_FOLDER = r"D:\Projects\Python\ML-based KPI prediction for O-RAN networks\raw data"
DESTINATION_FOLDER = r"D:\Projects\Python\ML-based KPI prediction for O-RAN networks\raw data\all raw data"
# ==================

source = Path(SOURCE_FOLDER)
destination = Path(DESTINATION_FOLDER)

# Create destination folder if it doesn't exist
destination.mkdir(parents=True, exist_ok=True)

# Find all CSV files recursively
csv_files = sorted(source.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files.\n")

for index, csv_file in enumerate(csv_files, start=1):
    new_name = f"raw{index}.csv"
    destination_file = destination / new_name

    shutil.copy2(str(csv_file), str(destination_file))

    print(f"Moved: {csv_file}")
    print(f"   -> {destination_file}\n")

print(f"Done! Moved {len(csv_files)} files.")