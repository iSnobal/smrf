.PHONY: clean-pyc clean-build clean-cython clean help tests tests-fast-fail tests-no-build install install-dev build_extensions

help: ## Display this help screen
	@sed -n 's/^\([A-Za-z0-9_.-]*\):.*## \(.*\)$$/\1:\2/p' $(MAKEFILE_LIST) | sort | column -t -s ':'

build_extensions: clean ## Build Cython extensions
	python setup.py build_ext --inplace

clean: clean-build clean-pyc ## Remove all build, test, coverage and Python artifacts

clean-cython: ## Clean up Cython generated C files
	find . -name detrended_kriging.c -type f -delete
	find . -name envphys_c.c -type f -delete
	find . -name wind_c.c -type f -delete

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.so' -type f -delete

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

tests: build_extensions ## Run tests and build extensions if necessary
	python3 -m unittest discover -s smrf/tests/

tests_fast_fail: build_extensions ## Run tests and stop at first failure
	python3 -m unittest discover -s smrf/tests/ -f

tests-no-build: ## Run tests without building extensions
	python3 -m unittest discover -s smrf/tests/

install: clean build_extensions ## Install the package with dependencies
	python3 -m pip install .[dev]

install-dev: clean ## Install the package without dependencies for development
	python3 -m pip install --no-deps -e .[dev]
