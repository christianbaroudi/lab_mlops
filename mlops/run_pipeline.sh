#!/bin/bash

set -e

echo "=== Preprocessing ==="
uv run mlops preprocess \
    --train ../resources/data/train.csv \
    --test ../resources/data/test.csv \
    --output ../resources/processed.csv

echo "=== Feature Engineering ==="
uv run mlops featurize \
    --input ../resources/processed.csv \
    --output ../resources/featurized.csv

echo "=== Training ==="
uv run mlops train \
    --input ../resources/featurized.csv

echo "=== Evaluation ==="
uv run mlops evaluate \
    --input ../resources/featurized.csv \
    --output ../resources/eval/

echo "=== Prediction ==="
uv run mlops predict \
    --input ../resources/featurized.csv \
    --output ../resources/predictions/

echo "=== Pipeline completed successfully ==="