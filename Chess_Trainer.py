import io
import zstandard as zstd
from pathlib import Path
import chess.pgn
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
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
    games = []
    for line in read_lines(zstd_file_path):
        pgn = io.StringIO(line)
        game = chess.pgn.read_game(pgn)
        if game is not None:
            result = game.headers.get("Result")
            white_player = game.headers.get("White", "?")
            black_player = game.headers.get("Black", "?")
            moves = " ".join(str(move) for move in game.mainline_moves())
            # Only add games with a winner and moves
            if moves and result in {"1-0", "0-1"}:
                game_text = f"White: {white_player}\nBlack: {black_player}\nResult: {result}\nMoves: {moves}"
                games.append({"text": game_text})
    return games

# Load and prepare data
file_path = Path('lichess_db_standard_rated_2022-02.pgn.zst')
chess_data = extract_chess_data(file_path)
dataset = Dataset.from_list(chess_data)

# Initialize tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("llama3.2")
model = LlamaForCausalLM.from_pretrained("llama3.2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./chess_llama",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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
