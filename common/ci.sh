#!/usr/bin/env bash

set -xueo pipefail

pip install -r requirements-tests.txt
flake8 models
yapf --recursive --diff models tests
pytest
