import io
import zstandard as zstd
from pathlib import Path
import chess.pgn
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import sentencepiece
from datasets import Dataset


DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

# Function to read lines from compressed file
def read_lines(zstd_file_path: Path):
    with (
        zstd.open(zstd_file_path, mode='rb', dctx=DCTX) as zfh,
        io.TextIOWrapper(zfh) as iofh
    ):
        for line in iofh:
            yield line


# Parse PGN data and format as text for LLaMA training
def extract_chess_data(zstd_file_path: Path):
    count = 0
    games = []
    for line in read_lines(zstd_file_path):
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if count == 999:
            break
        if game is not None and count < 1000:
            result = game.headers.get("Result")
            white_player = game.headers.get("White", "?")
            black_player = game.headers.get("Black", "?")
            moves = " ".join(str(move) for move in game.mainline_moves())
            # Only add games with a winner and moves
            if moves and result in {"1-0", "0-1"}:
                game_text = f"White: {white_player}\nBlack: {black_player}\nResult: {result}\nMoves: {moves}"
                games.append({"text": game_text})
                print(game_text)
                print(count)
                count +=1
    return games

# Load and prepare data
file_path = Path('lichess_db_standard_rated_2022-02.pgn.zst')
chess_data = extract_chess_data(file_path)
dataset = Dataset.from_list(chess_data)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the text and create labels identical to input_ids
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./chess_llama",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("fine_tuned_chess_llama")
tokenizer.save_pretrained("fine_tuned_chess_llama")

print("Training completed and model saved.")
