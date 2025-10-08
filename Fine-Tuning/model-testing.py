import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from validator import LayoutValidator

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./qwen-layout-optimizer-final")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, "./qwen-layout-optimizer-final")
model.eval()

# Load test input
with open('layouts/room-layout-multiple.json') as f:
    input_layout = json.load(f)

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

# Format prompt
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\n=== Generating optimized layout ===")

# Get input length to extract only new tokens
input_length = inputs['input_ids'].shape[1]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode only the NEW tokens (assistant's response)
generated_tokens = outputs[0][input_length:]
assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\n=== ASSISTANT RESPONSE ===")
print(assistant_response)

# Parse JSON
try:
    output_layout = json.loads(assistant_response)
    
    print("\n=== PARSED OUTPUT ===")
    print(json.dumps(output_layout, indent=2))
    
    # Save directly
    with open('model_output.json', 'w') as f:
        json.dump(output_layout, f, indent=2)
    print("\nOK Saved to model_output.json")
    
    # Validate
    validator = LayoutValidator(output_layout)
    violations = validator.validate()
    print(f"\n=== VALIDATION ===")
    print(f"Violations: {len(violations)}")
    if violations:
        for v in violations[:10]:
            print(f"  - {v}")
    else:
        print("  Perfect layout! Zero violations.")
    
except json.JSONDecodeError as e:
    print(f"\nJSON Parse Error: {e}")
    print("The model did not output valid JSON.")
    print("\nRaw output:")
    print(assistant_response)