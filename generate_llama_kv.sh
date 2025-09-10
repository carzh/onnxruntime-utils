#!/bin/bash

# Base paths
PYTHON_SCRIPT="/workspace/ort-forest/ort-pr/tools/python/onnx_test_data_utils.py"
INPUT_FILE="/workspace/onnxruntime-utils/past_key_values.npy"
OUTPUT_DIR="/workspace/models/llama_test1/test_input_1"

# Loop through layers 0-31
for i in {0..31}; do
    # Calculate output file index (starts at input_2.pb for layer 0)
    output_index=$((2 + i * 2))
    
    echo "Processing layer $i..."
    
    # Generate key file
    echo "  Creating key file: input_${output_index}.pb"
    python "$PYTHON_SCRIPT" \
        --action numpy_to_pb \
        --input "$INPUT_FILE" \
        --name "past_key_values.${i}.key" \
        --output "${OUTPUT_DIR}/input_${output_index}.pb"
    
    # Generate value file
    value_output_index=$((output_index + 1))
    echo "  Creating value file: input_${value_output_index}.pb"
    python "$PYTHON_SCRIPT" \
        --action numpy_to_pb \
        --input "$INPUT_FILE" \
        --name "past_key_values.${i}.value" \
        --output "${OUTPUT_DIR}/input_${value_output_index}.pb"
    
    echo "  Layer $i complete"
done

echo "All key-value pairs generated!"
