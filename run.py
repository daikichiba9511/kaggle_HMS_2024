import subprocess


def run_cmd(cmd: list[str]) -> None:
    _cmd = " ".join(cmd)
    print(f"Running: {_cmd}")
    subprocess.run(cmd, check=True)


exp_configs = [
    # ("exp029", [0, 1, 2]),
    # ("exp030", [0, 1, 2]),
    # ("exp031", [0, 1, 2]),
    # ("exp032", [0, 1, 2]),
    # ("exp033", [0, 1, 2]),
    # ("exp034", [0, 1, 2]),
    # ("exp035", [0, 1, 2]),
    # ("exp036", [0, 1, 2]),
    # ("exp037", [0, 1, 2]),
    # ("exp038", [0, 1, 2]),
    # ("exp039", [0, 1, 2]),
    # ("exp039", [0]),
    # ("exp040", [0, 1, 2]),
    # ("exp041", [0, 1, 2]),
    # ("exp042", [0, 1, 2]),
    ("exp043", [0, 1, 2]),
    # ("exp044", [0, 1, 2]),
]

for exp_ver, train_folds in exp_configs:
    print("Scheduled:", exp_ver, train_folds)

skipped_exp = []
for exp_ver, train_folds in exp_configs:
    run_cmd(["python", f"src/exp/{exp_ver}/train.py", "--train-folds", *map(str, train_folds)])
    run_cmd(["python", "scripts/valid.py", "--exp_ver", exp_ver[3:]])

print(f"Skipped: {skipped_exp}")
