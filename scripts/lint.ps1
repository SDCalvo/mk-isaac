# PowerShell script to run linting tools

# Format code with black
Write-Host "Formatting code with black..."
pipenv run black .

# Sort imports with isort
Write-Host "Sorting imports with isort..."
pipenv run isort .

# Lint with flake8
Write-Host "Linting with flake8..."
pipenv run flake8 src

# Remove trailing whitespace (PowerShell version)
Write-Host "Removing trailing whitespace..."
Get-ChildItem -Path . -Filter "*.py" -Recurse | ForEach-Object {
    (Get-Content $_.FullName) | ForEach-Object {
        $_ -replace '\s+$', ''
    } | Set-Content $_.FullName
}

Write-Host "Linting complete." 