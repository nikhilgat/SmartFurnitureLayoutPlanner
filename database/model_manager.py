import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root) 

import json
import torch
import threading
import time
from queue import Queue
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from FineTuning.validator import LayoutValidator

class ModelManager:
    def __init__(self, model_path="./qwen-layout-optimizer-version-1"):
        """Initialize the model manager"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model components
        self.tokenizer = None
        self.model = None
        
        # Loading status
        self.is_loading = False
        self.is_loaded = False
        self.loading_progress = 0  # 0-100
        self.loading_stage = "Initializing..."
        self.loading_error = None
        
        # Optimization queue
        self.optimization_queue = Queue()
        self.active_optimizations = {}  # {job_id: job_info}
        self.completed_optimizations = {}  # {job_id: result}
        
        # Background threads
        self.loader_thread = None
        self.worker_thread = None
        
    def start_loading(self):
        """Start loading the model in a background thread"""
        if self.is_loading or self.is_loaded:
            return
        
        self.is_loading = True
        self.loading_progress = 0
        self.loading_stage = "Starting model load..."
        
        self.loader_thread = threading.Thread(target=self._load_model, daemon=True)
        self.loader_thread.start()
        
        # Start optimization worker
        self.worker_thread = threading.Thread(target=self._optimization_worker, daemon=True)
        self.worker_thread.start()
    
    def _load_model(self):
        """Load the model (runs in background thread)"""
        try:
            print("\n" + "="*60)
            print("LOADING MODEL - Please wait...")
            print("="*60)
            
            # Stage 1: Device setup
            self.loading_stage = "Setting up device..."
            self.loading_progress = 10
            print(f"\n[DEVICE] Using: {self.device}")
            if self.device == "cuda":
                print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
            time.sleep(0.5)  # Small delay for UI update
            
            # Stage 2: Load tokenizer
            self.loading_stage = "Loading tokenizer..."
            self.loading_progress = 25
            print("\n[1/3] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            time.sleep(0.5)
            
            # Stage 3: Load base model
            self.loading_stage = "Loading base model (this may take a while)..."
            self.loading_progress = 40
            print("[2/3] Loading base model...")
            
            # Clear cache before loading
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Math-7B-Instruct",
                device_map="auto",
                torch_dtype=torch.float16,  # Always use float16 for GPU
                low_cpu_mem_usage=True,
                max_memory={0: "15GB", "cpu": "30GB"}  # Adjust based on your GPU
            )
            self.loading_progress = 60
            time.sleep(0.5)
            
            # Stage 4: Load adapter
            self.loading_stage = "Loading fine-tuned adapter..."
            self.loading_progress = 70
            print("[3/3] Loading fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            time.sleep(0.5)
            
            # Stage 5: Optimize model
            self.loading_stage = "Merging adapter weights..."
            self.loading_progress = 80
            print("[OPTIMIZATION] Merging adapter weights...")
            self.model = self.model.merge_and_unload()
            time.sleep(0.5)
            
            # Stage 6: Move to device
            self.loading_stage = "Moving model to device..."
            self.loading_progress = 90
            print(f"[OPTIMIZATION] Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            self.model.eval()
            time.sleep(0.5)
            
            # Complete
            self.loading_stage = "Model ready!"
            self.loading_progress = 100
            self.is_loaded = True
            self.is_loading = False
            
            print("\n" + "="*60)
            print("MODEL READY!")
            print("="*60 + "\n")
            
        except Exception as e:
            self.loading_error = str(e)
            self.is_loading = False
            self.loading_stage = f"Error: {str(e)}"
            print(f"\n[ERROR] Failed to load model: {e}")
    
    def get_loading_status(self):
        """Get current loading status"""
        return {
            'is_loading': self.is_loading,
            'is_loaded': self.is_loaded,
            'progress': self.loading_progress,
            'stage': self.loading_stage,
            'error': self.loading_error,
            'device': self.device
        }
    
    def submit_optimization(self, layout_data, original_layout_id):
        """
        Submit a layout for optimization
        
        Args:
            layout_data (dict): The layout to optimize
            original_layout_id (int): Database ID of the original layout
            
        Returns:
            str: Job ID for tracking
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded yet")
        
        # Generate unique job ID
        job_id = f"opt_{original_layout_id}_{int(time.time()*1000)}"
        
        # Create job info
        job_info = {
            'job_id': job_id,
            'layout_data': layout_data,
            'original_layout_id': original_layout_id,
            'status': 'queued',
            'progress': 0,
            'submitted_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'error': None
        }
        
        # Add to queue and tracking
        self.active_optimizations[job_id] = job_info
        self.optimization_queue.put(job_info)
        
        return job_id
    
    def get_optimization_status(self, job_id):
        """Get the status of an optimization job"""
        if job_id in self.completed_optimizations:
            return self.completed_optimizations[job_id]
        elif job_id in self.active_optimizations:
            return self.active_optimizations[job_id]
        else:
            return {'status': 'not_found', 'error': 'Job ID not found'}
    
    def _optimization_worker(self):
        """Background worker that processes optimization queue"""
        while True:
            # Wait for model to load
            if not self.is_loaded:
                time.sleep(1)
                continue
            
            # Get next job from queue
            job_info = self.optimization_queue.get()
            job_id = job_info['job_id']
            
            try:
                # Update status
                job_info['status'] = 'processing'
                job_info['progress'] = 10
                job_info['started_at'] = datetime.now().isoformat()
                
                print(f"\n[OPTIMIZATION] Starting job: {job_id}")
                start_time = time.time()
                
                # Run optimization
                job_info['progress'] = 30
                output_layout, error = self._optimize_layout(job_info['layout_data'])
                
                if error:
                    # Optimization failed
                    job_info['status'] = 'failed'
                    job_info['error'] = error
                    job_info['progress'] = 100
                    print(f"[OPTIMIZATION] Job {job_id} failed: {error}")
                else:
                    # Optimization successful
                    job_info['progress'] = 70
                    
                    # Validate the output
                    print(f"[OPTIMIZATION] Validating output...")
                    violations = self._validate_layout(output_layout)
                    
                    job_info['progress'] = 90
                    job_info['status'] = 'completed'
                    job_info['output_layout'] = output_layout
                    job_info['violations'] = violations
                    job_info['violations_count'] = len(violations)
                    job_info['progress'] = 100
                    
                    elapsed = time.time() - start_time
                    job_info['optimization_time'] = elapsed
                    
                    print(f"[OPTIMIZATION] Job {job_id} completed in {elapsed:.2f}s")
                    print(f"[OPTIMIZATION] Violations: {len(violations)}")
                
            except Exception as e:
                job_info['status'] = 'failed'
                job_info['error'] = str(e)
                job_info['progress'] = 100
                print(f"[OPTIMIZATION] Job {job_id} exception: {e}")
            
            finally:
                job_info['completed_at'] = datetime.now().isoformat()
                
                # Move to completed
                self.completed_optimizations[job_id] = job_info
                if job_id in self.active_optimizations:
                    del self.active_optimizations[job_id]
                
                # Mark queue task as done
                self.optimization_queue.task_done()
    
    def _optimize_layout(self, input_layout):
        """
        Generate optimized layout from input layout
        
        Args:
            input_layout (dict): The input layout
            
        Returns:
            tuple: (output_layout, error_message)
        """
        try:
            # Create prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert interior designer specializing in DIN 18040-2 compliant accessible bedroom layouts. Given an input layout, optimize furniture positions to comply with accessibility standards while maintaining livability."
                },
                {
                    "role": "user",
                    "content": json.dumps(input_layout)
                }
            ]
            
            # Format and tokenize
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Explicitly move inputs to the same device as model
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]
            
            # Generate with optimizations
            with torch.no_grad():
                # Use torch.cuda.amp for mixed precision if on GPU
                if self.device == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=1,
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                    )
            
            # Extract only generated tokens
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Parse JSON response
            try:
                output_layout = json.loads(response)
                return output_layout, None
            except json.JSONDecodeError as e:
                return None, f"JSON Parse Error: {e}\nRaw output: {response[:200]}..."
                
        except Exception as e:
            return None, f"Optimization error: {str(e)}"
    
    def _validate_layout(self, layout):
        """
        Validate layout and return violations
        
        Args:
            layout (dict): The layout to validate
            
        Returns:
            list: List of violation strings
        """
        try:
            validator = LayoutValidator(layout)
            violations = validator.validate()
            return violations
        except Exception as e:
            return [f"Validation error: {e}"]
    
    def get_queue_status(self):
        """Get current queue status"""
        return {
            'queue_size': self.optimization_queue.qsize(),
            'active_jobs': len(self.active_optimizations),
            'completed_jobs': len(self.completed_optimizations),
            'active_job_ids': list(self.active_optimizations.keys()),
            'completed_job_ids': list(self.completed_optimizations.keys())
        }
    
    def clear_completed_jobs(self, older_than_minutes=60):
        """Clear old completed jobs from memory"""
        cutoff_time = time.time() - (older_than_minutes * 60)
        
        to_remove = []
        for job_id, job_info in self.completed_optimizations.items():
            completed_timestamp = datetime.fromisoformat(job_info['completed_at']).timestamp()
            if completed_timestamp < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.completed_optimizations[job_id]
        
        return len(to_remove)


# Singleton instance
_model_manager_instance = None

def get_model_manager(model_path="./qwen-layout-optimizer-version-1"):
    """Get or create the singleton ModelManager instance"""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager(model_path)
    
    return _model_manager_instance


# Testing
if __name__ == '__main__':
    print("Testing ModelManager...")
    
    manager = get_model_manager()
    
    # Start loading
    print("Starting model load...")
    manager.start_loading()
    
    # Monitor loading
    while not manager.is_loaded and not manager.loading_error:
        status = manager.get_loading_status()
        print(f"Progress: {status['progress']}% - {status['stage']}")
        time.sleep(2)
    
    if manager.loading_error:
        print(f"Loading failed: {manager.loading_error}")
    else:
        print("Model loaded successfully!")
        
        # Test optimization
        test_layout = {
            "room": {"width": 400, "height": 300},
            "furniture": [
                {"name": "Bed", "x": 50, "y": 50, "width": 160, "height": 200, "zHeight": 55, "rotation": 0}
            ],
            "openings": []
        }
        
        print("\nSubmitting test optimization...")
        job_id = manager.submit_optimization(test_layout, original_layout_id=1)
        print(f"Job ID: {job_id}")
        
        # Monitor optimization
        while True:
            status = manager.get_optimization_status(job_id)
            print(f"Status: {status['status']} - Progress: {status.get('progress', 0)}%")
            
            if status['status'] in ['completed', 'failed']:
                print(f"\nFinal status: {status}")
                break
            
            time.sleep(2)