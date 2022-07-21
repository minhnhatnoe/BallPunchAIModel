from os.path import realpath, join, pardir
import subprocess


tolerance_exe = realpath(join(__file__, pardir, "solve_tolerance.exe"))
data_file = realpath(join(__file__, pardir, "data.txt"))

def transform(csv_path: str, data) -> None:
    with open(data_file, "w") as f:
        for row in data:
            f.write(f"{row[0]} {row[1]}\n")
    subprocess.run([tolerance_exe, data_file, csv_path])
