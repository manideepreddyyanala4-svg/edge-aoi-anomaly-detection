import subprocess
import sys


categories = ["bottle", "carpet", "grid", "capsule", "transistor"]


for category in categories:
    print(f"\nRunning {category}...\n")
    subprocess.run(
        [sys.executable, "-m", "src.build_memory", "--category", category],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--category", category],
        check=True,
    )
