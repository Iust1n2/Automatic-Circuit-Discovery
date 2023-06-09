import subprocess
import argparse
import os
from pathlib import Path
from experiments.launcher import KubernetesJob, WandbIdentifier, launch
import shlex
import random

IS_ADRIA = "arthur" not in __file__ and not __file__.startswith("/root")
print("is adria:", IS_ADRIA)

#TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion", "induction"]
TASKS = ["ioi", "docstring", "greaterthan", "induction"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["kl_div", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


def main(
    alg: str, 
    task: str, 
    job: KubernetesJob, 
    testing: bool = False, 
    mod_idx=0,
    num_processes=1,
):
    # mod_idx= MPI.COMM_WORLD.Get_rank()
    # num_processes = MPI.COMM_WORLD.Get_size()

    if IS_ADRIA:
        OUT_RELPATH = Path(".cache") / "plots_data"
        OUT_HOME_DIR = Path(os.environ["HOME"]) / OUT_RELPATH
    else:
        OUT_RELPATH = Path("experiments/results/arthur_plots_data") # trying to remove extra things from acdc/
        OUT_HOME_DIR = OUT_RELPATH

    assert OUT_HOME_DIR.exists()

    if IS_ADRIA:
        OUT_DIR = Path("/root") / OUT_RELPATH
    else:
        OUT_DIR = OUT_RELPATH

    seed = 1233778640
    random.seed(seed)

    commands = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for metric in METRICS_FOR_TASK[task]:
                command = [
                    "python",
                    "notebooks/roc_plot_generator.py",
                    f"--task={task}",
                    f"--reset-network={reset_network}",
                    f"--metric={metric}",
                    f"--alg={alg}",
                    f"--device={'cpu' if testing or not job.gpu else 'cuda'}",
                    f"--torch-num-threads={job.cpu}",
                    f"--out-dir={OUT_DIR}",
                    f"--seed={random.randint(0, 2**31-1)}",
                ]
                if zero_ablation:
                    command.append("--zero-ablation")

                if alg == "acdc" and task == "greaterthan" and metric == "kl_div" and not zero_ablation and not reset_network:
                    command.append("--ignore-missing-score")
                commands.append(command)

    if IS_ADRIA:
        launch(
            commands,
            name="collect_data",
            job=job,
            synchronous=True,
            just_print_commands=False,
            check_wandb=WandbIdentifier(f"agarriga-collect-{alg}-{task[-5:]}-{{i:05d}}", "collect", "acdc"),
        )

    else:
        for command_idx in range(mod_idx, len(commands), num_processes): # commands:
            # run 4 in parallel
            command = commands[command_idx]
            print(f"Running command {command_idx} / {len(commands)}")
            print(" ".join(command))
            subprocess.run(command)


tasks_for = {
    "acdc": TASKS,
    "16h": TASKS,
    "sp": TASKS,
}

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=int, default=0)
parser.add_argument("--n", type=int, default=1)

mod_idx = parser.parse_args().i
num_processes = parser.parse_args().n

if __name__ == "__main__":
    for alg in ["acdc", "16h", "sp"]:
        for task in tasks_for[alg]:
            main(
                alg,
                task,
                KubernetesJob(
                    container="ghcr.io/rhaps0dy/automatic-circuit-discovery:1.9.1",
                    cpu=4,
                    gpu=0 if not IS_ADRIA or task.startswith("tracr") or alg != "acdc" else 1,
                    mount_training=False,
                ),
                testing=False,
                mod_idx=mod_idx,
                num_processes=num_processes,
            )
