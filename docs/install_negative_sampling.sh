#!/bin/bash
# Quick installation script for negative sampling module
# Works around poetry lock file issue

set -e

echo "=========================================="
echo "Negative Sampling Module Installation"
echo "=========================================="
echo

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Function to check if package is installed
check_package() {
    python -c "import $1" 2>/dev/null && echo "✓ $1 is installed" || echo "✗ $1 is not installed"
}

# Check current status
echo
echo "Current dependency status:"
check_package pydantic
check_package pandas
check_package google.cloud.bigquery

echo
echo "Installation options:"
echo "  1. Minimal (negative sampling only)"
echo "  2. With BigQuery support (recommended)"
echo "  3. Skip installation (dependencies already satisfied)"
echo

read -p "Select option [1-3]: " option

case $option in
    1)
        echo
        echo "Installing minimal dependencies..."
        pip install pydantic
        ;;
    2)
        echo
        echo "Installing with BigQuery support..."
        pip install pydantic google-cloud-bigquery pandas
        ;;
    3)
        echo
        echo "Skipping installation..."
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo
echo "=========================================="
echo "Verification"
echo "=========================================="
echo

# Test imports
echo "Testing imports..."
python -c "import sys; sys.path.insert(0, 'src'); from legal_graph.processors.negative_sampling.strategies import SamplingStrategy; print('✅ Strategies module: OK')" || echo "❌ Strategies module: FAILED"

# Can only test main module if pydantic is available
if python -c "import pydantic" 2>/dev/null; then
    echo "Note: Full module testing requires pydantic to be installed"
    echo "      Run standalone tests with: python test_negative_sampling_simple.py"
fi

echo
echo "=========================================="
echo "Quick Test"
echo "=========================================="
echo

# Run standalone test
if [ -f "test_negative_sampling_simple.py" ]; then
    echo "Running standalone tests (no dependencies required)..."
    python test_negative_sampling_simple.py
else
    echo "Test file not found: test_negative_sampling_simple.py"
fi

echo
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Run tests:     python test_negative_sampling_simple.py"
echo "  2. View examples: python bigquery_integration_example.py"
echo "  3. Show CLI:      python -m legal_graph.processors.negative_sampling strategies"
echo
echo "Documentation:"
echo "  - README.md:      src/legal_graph/processors/negative_sampling/README.md"
echo "  - BigQuery Guide: BIGQUERY_INTEGRATION_GUIDE.md"
echo "  - Dependencies:   DEPENDENCIES.md"
echo