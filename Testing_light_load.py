# fine_tune_tinyllama.py

# Install required libraries (uncomment if not already installed)
# !pip install transformers datasets accelerate bitsandbytes peft

import bitsandbytes
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline,
)
from peft import LoraConfig, get_peft_model, PeftModel

def main():
    # Step 1: Load and Preprocess the Data
    # Replace 'Chess_training_data.json' with the path to your data file
    data_files = {'train': 'Chess_training_data.json'}
    dataset = load_dataset('json', data_files=data_files)

    # Step 2: Convert Moves to Text
    def preprocess_function(examples):
        return {
            'text': [
                ' '.join(moves) + f" Result: {result}"
                for moves, result in zip(examples['moves'], examples['result'])
            ]
        }

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )

    # Step 3: Initialize the Model and Tokenizer with 8-bit Quantization
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,  # Enable 8-bit loading with bitsandbytes
    )

    # Apply PEFT (LoRA)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Ensure model parallelism is enabled
    model.is_parallelizable = True
    model.model_parallel = True

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Step 4: Tokenize the Dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=256,  # Reduced sequence length
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=['text']
    )

    print(f"Number of training samples: {len(tokenized_dataset['train'])}")

    # Test data loading
    print("Testing data loading...")
    for idx, sample in enumerate(tokenized_dataset['train']):
        if idx >= 5:
            break
        print(f"Sample {idx}: {sample}")
    print("Data loading test completed.")

    # Step 5: Prepare Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Step 6: Set Up Training Arguments
    training_args = TrainingArguments(
        output_dir="./tinyllama-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=16,  # Increased accumulation steps
        evaluation_strategy="no",  # Corrected parameter name
        save_strategy="epoch",
        logging_steps=500,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",  # Optimizer compatible with 8-bit training
        dataloader_num_workers=0,
    )

    # Step 7: Create Custom Trainer to Prevent Moving the Model
    class CustomTrainer(Trainer):
        def _move_model_to_device(self, model, device):
            pass  # Override to prevent moving the model

    # Step 8: Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
    )

    # Step 9: Start Training
    trainer.train()

    # Step 10: Save the Fine-Tuned Model
    model.save_pretrained("./tinyllama-finetuned")
    tokenizer.save_pretrained("./tinyllama-finetuned")

    print(
        "Training complete. The fine-tuned model has been saved to './tinyllama-finetuned'."
    )

    # Step 11: Use the Fine-Tuned Model
    test_model()

def test_model():
    # Load the fine-tuned model
    model_path = "./tinyllama-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the base model with 8-bit precision
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_8bit=True,
    )

    # Load the PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)

    # Create the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chess bot who is incredible at predicting the best next move "
                "based off the current board. Please respond in PGN and tell me where the "
                "piece starts and where it should end."
            ),
        },
        {
            "role": "user",
            "content": (
                "What should I do based off of this board? "
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            ),
        },
    ]

    # Generate the prompt
    prompt = (
        f"System: {messages[0]['content']}\n"
        f"User: {messages[1]['content']}\n"
        "Assistant:"
    )

    # Generate the response
    outputs = pipe(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    print("Model Output:")
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()
