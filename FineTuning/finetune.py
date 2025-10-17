from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Model name
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Preparing model for training...")
model = prepare_model_for_kbit_training(model)

# IMPROVED LoRA config
lora_config = LoraConfig(
    r=64,  # Increased from 16
    lora_alpha=128,  # Increased proportionally (2*r)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset('json', data_files='finetuning_data.jsonl', split='train')

# SPLIT INTO TRAIN/VAL
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train examples: {len(train_dataset)}")
print(f"Eval examples: {len(eval_dataset)}")

def tokenize_function(examples):
    texts = []
    for messages in examples['messages']:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=2048
    )
    
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

print("Tokenizing dataset...")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

# IMPROVED Training args
training_args = TrainingArguments(
    output_dir="./qwen-layout-optimizer",
    num_train_epochs=5,  # Reduced from 10
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Increased from 4
    learning_rate=1e-4,  # Reduced from 2e-4
    lr_scheduler_type="cosine",  # Added scheduler
    warmup_steps=100,  # Added warmup
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",  # Added evaluation
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,  # Keep only best 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_8bit",
    report_to="none"  # Disable wandb/tensorboard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Added
    tokenizer=tokenizer
)

print("\n=== Starting Fine-Tuning ===")
trainer.train()

print("\n=== Saving Model ===")
model.save_pretrained("./qwen-layout-optimizer-final")
tokenizer.save_pretrained("./qwen-layout-optimizer-final")

print("OK Fine-tuning complete!")