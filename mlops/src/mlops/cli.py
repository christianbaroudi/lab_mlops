"""Command line entry point: uv run mlops <step> [args]"""

import argparse

from mlops.eval import evaluate
from mlops.featurize import featurize
from mlops.predict import predict
from mlops.preprocess import preprocess
from mlops.train import MODELS, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlops", description="Titanic survival pipeline.")
    steps = parser.add_subparsers(dest="step", required=True)

    p = steps.add_parser("preprocess", help="clean train.csv and test.csv and merge them")
    p.add_argument("--train", required=True, help="raw train.csv")
    p.add_argument("--test", required=True, help="raw test.csv")
    p.add_argument("--output", default="resources/processed.csv")

    p = steps.add_parser("featurize", help="add Title and Family_size")
    p.add_argument("--input", default="resource/processed.csv")
    p.add_argument("--output", default="resources/featurized.csv")

    p = steps.add_parser("train", help="train a model and save it to resource/models/<model>.pkl")
    p.add_argument("--input", default="resources/featurized.csv")

    p = steps.add_parser("evaluate", help="accuracy of a saved model on the held-out train rows")
    p.add_argument("--input", default="resources/featurized.csv")
    p.add_argument("--output", default="resources/eval/")

    p = steps.add_parser("predict", help="predict the test rows with a saved model")
    p.add_argument("--input", default="resources/featurized.csv")
    p.add_argument("--output", default="resources/predictions/")

    return parser


def main():
    args = build_parser().parse_args()

    if args.step == "preprocess":
        preprocess(args.train, args.test, args.output)
    elif args.step == "featurize":
        featurize(args.input, args.output)
    elif args.step == "train":
        train(args.input)
    elif args.step == "evaluate":
        evaluate(args.input, args.output)
    elif args.step == "predict":
        predict(args.input, args.output)


if __name__ == "__main__":
    main()
