#!/bin/bash

# Format code with black
echo "Formatting code with black..."
pipenv run black .

# Sort imports with isort
echo "Sorting imports with isort..."
pipenv run isort .

# Lint with flake8
echo "Linting with flake8..."
pipenv run flake8 src

# Remove trailing whitespace
echo "Removing trailing whitespace..."
find . -name "*.py" -exec sed -i 's/[ \t]*$//' {} \;

echo "Linting complete." 