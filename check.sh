#!/bin/bash

# Run Pytest
pytest
PYTEST_EXIT_CODE=$?

# Run mypy
mypy .
MYPY_EXIT_CODE=$?

# Run isort
isort --check .
ISORT_EXIT_CODE=$?

# Run Black
black --check .
BLACK_EXIT_CODE=$?

# Run flake8
flake8
FLAKE8_EXIT_CODE=$?

# Check if all tests and checks passed
if [ $PYTEST_EXIT_CODE -eq 0 ] && [ $MYPY_EXIT_CODE -eq 0 ] && [ $ISORT_EXIT_CODE -eq 0 ] && [ $BLACK_EXIT_CODE -eq 0 ] && [ $FLAKE8_EXIT_CODE -eq 0 ]; then
    echo "Success"
else
    echo "Tests/Checks failed"
    exit 1
fi

