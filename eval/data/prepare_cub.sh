#!/bin/bash

# Path to the images directory
IMAGES_DIR="data/CUB_200_2011/images"

# Create a temporary directory to store the flattened images
TEMP_DIR="${IMAGES_DIR}_flat"
mkdir -p "$TEMP_DIR"

# Loop through all subdirectories and move images to the temporary directory
for CLASS_DIR in "$IMAGES_DIR"/*; do
    if [ -d "$CLASS_DIR" ]; then
        CLASS_NAME=$(basename "$CLASS_DIR")
        for IMAGE_FILE in "$CLASS_DIR"/*; do
            if [ -f "$IMAGE_FILE" ]; then
                BASENAME=$(basename "$IMAGE_FILE")
                mv "$IMAGE_FILE" "$TEMP_DIR/${CLASS_NAME}_${BASENAME}"
            fi
        done
        rmdir "$CLASS_DIR"
    fi
done

# Move the flattened images back to the original images directory
mv "$TEMP_DIR"/* "$IMAGES_DIR"
rmdir "$TEMP_DIR"

echo "Flattened image directory structure."
