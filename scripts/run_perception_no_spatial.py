"""Same session-disjoint base+continuation recipe; change only the protocol data."""
import argparse
import json
import os
import shlex
import subprocess
import sys

from perception_no_spatial_protocols import ROOT, PROTOCOLS, STAGES, load_config, check_protocol


def commands(protocol, mode):
    train = lambda stage: [sys.executable, "scripts/train_perception_no_spatial.py", "--protocol", protocol, "--stage", stage]
    test = lambda split: [sys.executable, "scripts/test_perception_no_spatial.py", "--protocol", protocol, "--split", split]
    return {"all": [train("base"), train("continue"), test("all")],
            "train": [train("base"), train("continue")],
            "base": [train("base")], "continue": [train("continue")],
            "val": [test("val")], "test": [test("all")], "check": []}[mode]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--mode", choices=("all", "check", "train", "base", "continue", "val", "test"), default="all")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; otherwise preserve CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without training or writing outputs")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for stage in STAGES:
        load_config(args.protocol, stage)
    queue = commands(args.protocol, args.mode)
    if args.dry_run:
        for command in queue:
            print(shlex.join(command))
        return
    print(json.dumps(check_protocol(args.protocol), indent=2), flush=True)
    for command in queue:
        print("[run] " + shlex.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
