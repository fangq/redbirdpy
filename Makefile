# Redbird Makefile
# Common commands for building, testing, and publishing

.PHONY: help install dev test coverage lint clean build publish docs pretty

PYTHON := python3
PIP := pip3

help:
	@echo "Redbird - Diffuse Optical Tomography Toolbox"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install package"
	@echo "  make dev        - Install in development mode with dev dependencies"
	@echo "  make test       - Run unit tests"
	@echo "  make coverage   - Run tests with coverage report"
	@echo "  make lint       - Run code linting"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make build      - Build distribution packages"
	@echo "  make publish    - Publish to PyPI (requires credentials)"
	@echo "  make testpypi   - Publish to TestPyPI"
	@echo ""

install:
	$(PIP) install .

dev:
	$(PIP) install -e ".[dev,mesh]"

test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v -x --tb=short

coverage:
	$(PYTHON) -m pytest tests/ --cov=redbird --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	$(PYTHON) -m flake8 redbird/ tests/
	@echo "Linting complete"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"

build: clean
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build
	@echo "Built packages in dist/"
	@ls -la dist/

publish: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload dist/*

testpypi: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

# Check package before publishing
check: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine check dist/*

# Install from TestPyPI (for testing)
install-testpypi:
	$(PIP) install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ redbird

# Generate documentation (requires sphinx)
docs:
	cd docs && make html

# Format code (requires black)
format:
	$(PYTHON) -m black redbird/ tests/

# Type checking (requires mypy)
typecheck:
	$(PYTHON) -m mypy redbird/

# Run all checks before commit
precommit: lint test
	@echo "All checks passed!"

# Show package info
info:
	@$(PYTHON) -c "import redbird; redbird.info()"

# Show version
version:
	@$(PYTHON) -c "import redbird; print(redbird.__version__)"

pretty:
	$(PYTHON) -m black test/*.py redbird/*.py setup.py