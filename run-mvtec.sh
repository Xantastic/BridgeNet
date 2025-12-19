#!/bin/bash

# BridgeNet Run Script for MVTec-3D Dataset
# This script runs BridgeNet on sample categories from the MVTec-3D dataset

# Configuration
DATASET_PATH="/root/3D/GLASS-mvtec-3d-dataset/datasets/mvtec_process"
AUG_PATH="./data/aug"
RESULTS_PATH="./results"
CATEGORIES=("cookie" "dowel")
BACKBONE="wideresnet50"
LAYERS="layer2 layer3"

# Create augmentation directory if it doesn't exist
mkdir -p ${AUG_PATH}

# Function to run training for a single category
train_category() {
    local category=$1
    echo "==================================================="
    echo "Training on category: ${category}"
    echo "==================================================="

    python main.py \
        net -b ${BACKBONE} -le ${LAYERS} \
        dataset mvtec3d ${DATASET_PATH} ${AUG_PATH} \
        -d ${category} \
        --batch_size 8 \
        --meta_epochs 640 \
        --lr 0.00005 \
        --results_path ${RESULTS_PATH} \
        --run_name bridgenet_${category}
}

# Function to test a category
test_category() {
    local category=$1
    echo "==================================================="
    echo "Testing on category: ${category}"
    echo "==================================================="

    python main.py \
        net -b ${BACKBONE} -le ${LAYERS} \
        dataset mvtec3d ${DATASET_PATH} ${AUG_PATH} \
        -d ${category} \
        --test_mode png \
        --results_path ${RESULTS_PATH} \
        --run_name test_${category}
}

# Main execution
echo "BridgeNet Training Script for MVTec-3D"
echo "========================================"
echo "Dataset path: ${DATASET_PATH}"
echo "Categories: ${CATEGORIES[@]}"
echo "Results will be saved to: ${RESULTS_PATH}"
echo ""

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset path not found: ${DATASET_PATH}"
    echo "Please ensure the dataset is downloaded and the path is correct."
    exit 1
fi

# Run training for all categories
for category in "${CATEGORIES[@]}"; do
    # Check if category exists in dataset
    if [ ! -d "${DATASET_PATH}/${category}" ]; then
        echo "WARNING: Category '${category}' not found in dataset, skipping..."
        continue
    fi

    train_category ${category}
    echo ""
    echo "Training completed for ${category}"
    echo ""
done

echo "All training completed!"
echo "Results are saved in: ${RESULTS_PATH}"

# Optional: Run testing
read -p "Do you want to run testing? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for category in "${CATEGORIES[@]}"; do
        if [ -d "${DATASET_PATH}/${category}" ]; then
            test_category ${category}
            echo ""
        fi
    done
    echo "All testing completed!"
fi

echo "Done!"
