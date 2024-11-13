# fine_tune_tinyllama.py

# Install required libraries (uncomment if not already installed)
# !pip install transformers datasets accelerate

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
    pipeline
)

def main():
    # Step 1: Load and Preprocess the Data
    # Replace 'output.json' with the path to your data file
    data_files = {'train': 'Chess_training_data.json'}
    dataset = load_dataset('json', data_files=data_files)

    # Step 2: Convert Moves to Text
    def preprocess_function(examples):
        return {
            'text': [' '.join(moves) + f" Result: {result}" for moves, result in zip(examples['moves'], examples['result'])]
        }

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    # Step 3: Initialize the Model and Tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Step 4: Tokenize the Dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    print(f"Number of training samples: {len(tokenized_dataset['train'])}")

    # Add this code before the training loop to test data loading
    print("Testing data loading...")
    for idx, sample in enumerate(tokenized_dataset['train']):
        if idx >= 5:
            break
        print(f"Sample {idx}: {sample}")
    print("Data loading test completed.")

    # Step 5: Prepare Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Step 6: Set Up Training Arguments
    training_args = TrainingArguments(
        output_dir="./tinyllama-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=500,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,  # Added
    )

    # Step 7: Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Step 8: Start Training
    trainer.train()

    # Step 9: Save the Fine-Tuned Model
    trainer.save_model("./tinyllama-finetuned")
    tokenizer.save_pretrained("./tinyllama-finetuned")

    print("Training complete. The fine-tuned model has been saved to './tinyllama-finetuned'.")

    # Step 10: Use the Fine-Tuned Model
    # Optionally, you can test the model after training
    test_model()

def test_model():
    # Load the fine-tuned model
    model_path = "./tinyllama-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a chess bot who is incredible at predicting the best next move based off the current board. Please respond in PGN and tell me where the piece starts and where it should end.",
        },
        {
            "role": "user",
            "content": "What should I do based off of this board? rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

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
