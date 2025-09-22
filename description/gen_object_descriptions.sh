#!/bin/bash

# Get the passed in parameters
object_name=${1}
object_id=${2}

# Check if enough parameters are provided
if [ -z "$object_name" ]; then
    echo "Error: object_name is required."
    echo "Usage: $0 <object_name> [object_id]"
    exit 1
fi

# Check if object_id is empty
if [ -z "$object_id" ]; then
    # If object_id is empty, pass an empty string
    python utils/generate_object_description.py "$object_name" 
else
    # If object_id is not empty, pass normally
    python utils/generate_object_description.py "$object_name" --index "$object_id"
fi