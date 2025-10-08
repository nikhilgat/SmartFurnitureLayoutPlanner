import json
import os
import glob

def create_finetuning_dataset():
    """Convert JSON pairs to fine-tuning format"""
    
    training_examples = []
    
    # Read all example pairs
    # Count how many examples exist

    input_files = glob.glob('training_data/example_*_input.json')
    num_examples = len(input_files)

    print(f"Found {num_examples} training examples")

    # Read all example pairs
    for i in range(1, num_examples + 1):
        input_file = f'training_data/example_{i:03d}_input.json'
        output_file = f'training_data/example_{i:03d}_output.json'
        
        if not os.path.exists(input_file) or not os.path.exists(output_file):
            print(f"Skipping example {i} - files not found")
            continue
        
        with open(input_file) as f:
            input_layout = json.load(f)
        
        with open(output_file) as f:
            output_layout = json.load(f)
        
        # Create training example in Qwen format
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert interior designer specializing in DIN 18040-2 compliant accessible bedroom layouts. Given an input layout, optimize furniture positions to comply with accessibility standards while maintaining livability."
                },
                {
                    "role": "user",
                    "content": json.dumps(input_layout)
                },
                {
                    "role": "assistant",
                    "content": json.dumps(output_layout)
                }
            ]
        }
        
        training_examples.append(example)
        print(f"OK Processed example {i}")
    
    # Save as JSONL (one JSON object per line)
    with open('finetuning_data.jsonl', 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nOK Created finetuning_data.jsonl with {len(training_examples)} examples")
    print(f"OK Ready for fine-tuning")

if __name__ == '__main__':
    create_finetuning_dataset()