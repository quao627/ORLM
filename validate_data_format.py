#!/usr/bin/env python3
"""
Simple validation script to check if converted data has the correct format
without requiring the full transformers library.
"""

import json
import sys
from pathlib import Path


def validate_messages_format(data_file):
    """Validate that the data file has the correct messages format"""
    print(f"Validating messages format in {data_file}...")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return False
    
    if not isinstance(data, list):
        print("❌ Data should be a list of examples")
        return False
    
    if len(data) == 0:
        print("❌ Data list is empty")
        return False
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Check first few examples
    for i, example in enumerate(data[:3]):
        print(f"\nChecking example {i+1}:")
        
        # Check required fields
        if "messages" not in example:
            print(f"❌ Example {i+1}: Missing 'messages' field")
            return False
        
        messages = example["messages"]
        if not isinstance(messages, list):
            print(f"❌ Example {i+1}: 'messages' should be a list")
            return False
        
        if len(messages) == 0:
            print(f"❌ Example {i+1}: 'messages' list is empty")
            return False
        
        # Check message structure
        for j, message in enumerate(messages):
            if not isinstance(message, dict):
                print(f"❌ Example {i+1}, Message {j+1}: Message should be a dict")
                return False
            
            if "role" not in message:
                print(f"❌ Example {i+1}, Message {j+1}: Missing 'role' field")
                return False
            
            if "content" not in message:
                print(f"❌ Example {i+1}, Message {j+1}: Missing 'content' field")
                return False
            
            if message["role"] not in ["system", "user", "assistant"]:
                print(f"❌ Example {i+1}, Message {j+1}: Invalid role '{message['role']}'")
                return False
        
        print(f"✅ Example {i+1}: Valid messages format")
        
        # Show message structure
        print(f"   Messages: {len(messages)} messages")
        for j, msg in enumerate(messages):
            print(f"     {j+1}. {msg['role']}: {msg['content'][:50]}...")
    
    print(f"\n✅ All {len(data)} examples have valid messages format!")
    return True


def validate_prompt_completion_format(data_file):
    """Validate that the data file has the correct prompt/completion format"""
    print(f"Validating prompt/completion format in {data_file}...")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return False
    
    if not isinstance(data, list):
        print("❌ Data should be a list of examples")
        return False
    
    if len(data) == 0:
        print("❌ Data list is empty")
        return False
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Check first few examples
    for i, example in enumerate(data[:3]):
        print(f"\nChecking example {i+1}:")
        
        # Check required fields
        if "prompt" not in example:
            print(f"❌ Example {i+1}: Missing 'prompt' field")
            return False
        
        if "completion" not in example:
            print(f"❌ Example {i+1}: Missing 'completion' field")
            return False
        
        if not isinstance(example["prompt"], str):
            print(f"❌ Example {i+1}: 'prompt' should be a string")
            return False
        
        if not isinstance(example["completion"], str):
            print(f"❌ Example {i+1}: 'completion' should be a string")
            return False
        
        print(f"✅ Example {i+1}: Valid prompt/completion format")
        print(f"   Prompt: {example['prompt'][:50]}...")
        print(f"   Completion: {example['completion'][:50]}...")
    
    print(f"\n✅ All {len(data)} examples have valid prompt/completion format!")
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_data_format.py <data_file>")
        print("Example: python validate_data_format.py train_test_data/converted_data_messages.json")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    if not Path(data_file).exists():
        print(f"❌ File not found: {data_file}")
        sys.exit(1)
    
    # Try to determine format by checking the first example
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("❌ Empty data file")
            sys.exit(1)
        
        first_example = data[0]
        
        if "messages" in first_example:
            success = validate_messages_format(data_file)
        elif "prompt" in first_example and "completion" in first_example:
            success = validate_prompt_completion_format(data_file)
        else:
            print("❌ Unknown data format. Expected 'messages' or 'prompt'/'completion' fields")
            sys.exit(1)
        
        if success:
            print("\n🎉 Data format validation passed!")
            sys.exit(0)
        else:
            print("\n💥 Data format validation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
