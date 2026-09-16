"""Run one stage of the original Perception recipe on a held-out protocol."""
import argparse
import os

from perception_no_spatial_protocols import ROOT, PROTOCOLS, STAGES, train_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    train_stage(args.protocol, args.stage)


if __name__ == "__main__":
    main()
