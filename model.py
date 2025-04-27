from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AlbertTokenizer, AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from sacrebleu import corpus_bleu
from datasets import Dataset
import os
from tqdm import tqdm
from transformers import set_seed

set_seed(42)

data_path = "/raid/home/kshreya/data_aug_nmt/Dataset/English-Sanskrit/org_data" 

print("data_path ", data_path)

train_src = f"{data_path}/train.src"
train_tgt = f"{data_path}/train.tgt"
valid_src = f"{data_path}/dev.src"
valid_tgt = f"{data_path}/dev.tgt"
test_src = f"{data_path}/test.src"
test_tgt = f"{data_path}/test.tgt"

# 1. Load data
def load_data(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as src_file, open(tgt_path, "r", encoding="utf-8") as tgt_file:
        src_lines = [line.strip() for line in src_file.readlines()]
        tgt_lines = [line.strip() for line in tgt_file.readlines()]
    return {"src": src_lines, "tgt": tgt_lines}

train_data = load_data(train_src, train_tgt)
valid_data = load_data(valid_src, valid_tgt)
test_data = load_data(test_src, test_tgt)

# 2. Convert to Dataset format
train_dataset = Dataset.from_dict(train_data)
valid_dataset = Dataset.from_dict(valid_data)
test_dataset = Dataset.from_dict(test_data)

# 3. Load tokenizer and model
#model_name = "facebook/m2m100_418M"
#model = M2M100ForConditionalGeneration.from_pretrained(model_name)
#tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# ---1---
#tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
#model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# ---2---
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")


# Tokenize data
def preprocess_function(examples):
    inputs = tokenizer(examples["src"], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples["tgt"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_valid = valid_dataset.map(preprocess_function, batched=True)

# 4. Define training arguments
output_dir = "./checkpoints/en-san-nllb"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",  # Evaluate at the same interval as saving
    save_strategy="steps",        # Save checkpoints every 10,000 steps
    save_steps=500,             # Save every 10,000 steps
    eval_steps=500,             # Evaluate every 10,000 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,           # Keep only the last 3 checkpoints
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=500,            # Log progress every 500 steps
    load_best_model_at_end=True,  # Load the best model based on evaluation
    resume_from_checkpoint=True,  # Resume training if interrupted
)

# 5. Resume from the latest checkpoint if available
last_checkpoint = None
if os.path.exists(output_dir) and os.listdir(output_dir):
    last_checkpoint = max(
        [os.path.join(output_dir, d) for d in os.listdir(output_dir)],
        key=os.path.getctime,
    )
    print(f"Resuming from checkpoint: {last_checkpoint}")

# 6. Fine-tune the model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
)

trainer.train()
#trainer.train(resume_from_checkpoint=last_checkpoint)







