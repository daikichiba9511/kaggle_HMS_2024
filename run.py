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
    ("exp033", [0, 1, 2]),
    ("exp034", [0, 1, 2]),
    ("exp035", [0, 1, 2]),
    ("exp036", [0, 1, 2]),
    ("exp037", [0, 1, 2]),
    ("exp038", [0, 1, 2]),
]

skipped_exp = []
for exp_ver, train_folds in exp_configs:
    try:
        run_cmd(["python", f"src/exp/{exp_ver}/train.py", "--train-folds", *map(str, train_folds)])
        run_cmd(["python", "scripts/valid.py", "--exp_ver", exp_ver[3:]])
    except Exception as e:
        print(f"Error: {e}")
        skipped_exp.append(exp_ver)

print(f"Skipped: {skipped_exp}")
