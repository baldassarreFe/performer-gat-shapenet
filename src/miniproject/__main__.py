import argparse

from miniproject.train import add_subparser as add_subparser_train
from miniproject.test import add_subparser as add_subparser_test
from miniproject.configuration import add_subparser as add_subparser_conf

parser = argparse.ArgumentParser(
    prog="python -m miniproject", description="ShapeNet classifier"
)
subparsers = parser.add_subparsers()
add_subparser_train(subparsers)
add_subparser_test(subparsers)
add_subparser_conf(subparsers)
args = parser.parse_args()
args.cmd(**vars(args))
