"""Existing Perception evaluation: full validation calibration, then test."""
import argparse
import os

from perception_no_spatial_protocols import ROOT, PROTOCOLS, evaluate_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    args = parser.parse_args()
    os.chdir(ROOT)
    evaluate_protocol(args.protocol, args.split)


if __name__ == "__main__":
    main()
