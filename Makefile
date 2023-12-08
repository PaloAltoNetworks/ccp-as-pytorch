THIS_FILE := $(lastword $(MAKEFILE_LIST))
THIS_DIR:=$(shell dirname $(realpath $(THIS_FILE)))

.PHONY: format
format:
	cd ${THIS_DIR}; isort . --profile black
	cd ${THIS_DIR}; black --target-version py37 .

.PHONY: lint
lint:
	cd ${THIS_DIR}; python -m mypy .
	cd ${THIS_DIR}; python -m isort * --check-only --profile black
	cd ${THIS_DIR}; python -m flake8 .
	cd ${THIS_DIR}; python -m black --check .

.PHONY: strip
strip:
	cd ${THIS_DIR}; nbstripout usage-examples/text-ag-news/usage.ipynb

.PHONY: test
test:
	cd ${THIS_DIR}; pytest
