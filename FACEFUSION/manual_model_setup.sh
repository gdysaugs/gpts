#!/bin/bash

# Manual model setup for FaceFusion 3.3.0
# Since auto download is not working, we'll let FaceFusion download models on first run

echo "Setting up FaceFusion 3.3.0 models..."

# Create necessary directories
mkdir -p .assets/models
mkdir -p output
mkdir -p temp

echo "Model directories created."
echo "Models will be downloaded during first run of FaceFusion."
echo ""
echo "To test FaceFusion, run:"
echo "  ./face_swap_cli.sh input/source.jpg input/target.jpg"
echo ""
echo "Note: First run will take longer as models are downloaded automatically."