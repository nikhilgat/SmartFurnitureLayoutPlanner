import subprocess
import sys
import time
import os
from datetime import datetime

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open('pipeline.log', 'a', encoding='utf-8') as f:  # Add encoding='utf-8'
        f.write(msg + '\n')

def run_command(description, command):
    """Run a command and handle errors with real-time output"""
    log(f"{'='*60}")
    log(f"STARTING: {description}")
    log(f"Command: {command}")
    log(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run without capturing output - let it print in real-time
        result = subprocess.run(
            command,
            shell=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        log(f"OK COMPLETED: {description} (took {elapsed/60:.1f} minutes)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        log(f"FAILED: {description} (after {elapsed/60:.1f} minutes)")
        log(f"Exit code: {e.returncode}")
        return False
    
def main(num_examples):
    """Run complete pipeline"""
    
    log("\n" + "="*60)
    log(f"PIPELINE STARTED - Generating {num_examples} examples")
    log("="*60 + "\n")
    
    pipeline_start = time.time()
    
    # Step 1: Generate dataset
    if not run_command(
        f"Dataset Generation ({num_examples} examples)",
        f"python generate_dataset.py {num_examples}"
    ):
        log("Pipeline failed at dataset generation")
        return False
    
    # Step 2: Prepare fine-tuning data
    if not run_command(
        "Prepare Fine-Tuning Data",
        "python fine-tuning-dataset.py"
    ):
        log("Pipeline failed at data preparation")
        return False
    
    # Step 3: Fine-tune model
    if not run_command(
        "Model Fine-Tuning",
        "python finetune.py"
    ):
        log("Pipeline failed at fine-tuning")
        return False
    
    # Success!
    total_elapsed = time.time() - pipeline_start
    log("\n" + "="*60)
    log(f"OK PIPELINE COMPLETED SUCCESSFULLY")
    log(f"Total time: {total_elapsed/3600:.1f} hours")
    log("="*60 + "\n")
    
    return True

if __name__ == '__main__':
    # Get number of examples from command line or use default
    if len(sys.argv) > 1:
        num_examples = int(sys.argv[1])
    else:
        print("Usage: python run_pipeline.py <num_examples>")
        print("Example: python run_pipeline.py 5000")
        sys.exit(1)
    
    success = main(num_examples)
    sys.exit(0 if success else 1)