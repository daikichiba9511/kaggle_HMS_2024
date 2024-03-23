import subprocess


def run_cmd(cmd: list[str]) -> None:
    _cmd = " ".join(cmd)
    print(f"Running: {_cmd}")
    subprocess.run(cmd, check=True)


exp_configs = [
    # ("exp029", [0, 1, 2]),
    ("exp030", [0, 1, 2]),
    ("exp031", [0, 1, 2]),
    ("exp032", [0, 1, 2]),
    ("exp033", [0, 1, 2]),
    ("exp034", [0, 1, 2]),
]

for exp_ver, train_folds in exp_configs:
    run_cmd(["python", f"src/exp/{exp_ver}/train.py", "--train-folds", *map(str, train_folds)])
    run_cmd(["python", "scripts/valid.py", "--exp_ver", exp_ver[3:]])
