import numpy as np
import random
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator
import nltk
import os
nltk.download('punkt')
nltk.download('punkt_tab')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
ASSAMESE_TRAIN_FILE = "./Dataset/English-Hindi/org_data/train_trimmed_filtered.tgt"
MODEL_NAME = "facebook/nllb-200-3.3B"  # Changed to NLLB-200 model
#MODEL_SAVE_PATH = "assamese_rearrangement_model-nllb-final"
MAX_LENGTH = 128 #128
BATCH_SIZE = 32 #16
EPOCHS = 10 #5
NUM_PERMUTATIONS = 10 #10 # Number of permutations to generate per sentence

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

CHECKPOINT_DIR = "re-checkpoint-hindi"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Step 1: Prepare Training Data
class PermutationDataset(Dataset):
    def __init__(self, filename, tokenizer, num_permutations=5):
        self.tokenizer = tokenizer
        self.examples = []
        
        print(f"\nLoading data from {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for original_sentence in tqdm(lines, desc="Creating permutations"):
            original_tokens = nltk.word_tokenize(original_sentence)
            
            for _ in range(num_permutations):
                permuted_tokens = original_tokens.copy()
                random.shuffle(permuted_tokens)
                permuted_sentence = ' '.join(permuted_tokens)
                self.examples.append((permuted_sentence, original_sentence))
        
        print(f"Created {len(self.examples)} training examples\n")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def find_last_checkpoint(checkpoint_dir):
    """Find the highest epoch checkpoint directory"""
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('epoch_')]
    if not checkpoints:
        return None
    # Extract epoch numbers and find the max
    epoch_numbers = [int(chk.split('_')[1]) for chk in checkpoints]
    last_epoch = max(epoch_numbers)
    return os.path.join(checkpoint_dir, f'epoch_{last_epoch}')

# Check if we have any checkpoints to resume from
last_checkpoint = find_last_checkpoint(CHECKPOINT_DIR)

if last_checkpoint:
    print(f"\nResuming training from checkpoint: {last_checkpoint}")
    model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="hin_Deva", tgt_lang="hin_Deva")  # Assamese language code for NLLB
    
    # Determine the starting epoch (last completed epoch + 1)
    starting_epoch = int(os.path.basename(last_checkpoint).split('_')[1])
    print(f"Resuming training from epoch {starting_epoch + 1}/{EPOCHS}")
else:
    print("\nNo checkpoints found, starting training from scratch")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="hin_Deva", tgt_lang="hin_Deva")  # Assamese language code for NLLB
    starting_epoch = 0

if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        list(inputs),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            list(targets),
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        )["input_ids"]
    
    # Replace padding token id with -100 for loss calculation
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs

print("Creating data loaders...")
train_dataset = PermutationDataset(ASSAMESE_TRAIN_FILE, tokenizer, NUM_PERMUTATIONS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Step 4: Training Loop
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
    
    for batch in progress_bar:
        # Move all tensors in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        
        # Handle DataParallel output
        loss = outputs.loss
        if isinstance(loss, torch.Tensor):  # When using DataParallel
            loss = loss.mean()  # Average losses across GPUs
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

print("\nStarting training...")
for epoch in range(starting_epoch, EPOCHS):
    avg_loss = train_epoch(model, train_loader, optimizer, epoch)
    print(f"Epoch {epoch+1}/{EPOCHS} completed | Loss: {avg_loss:.4f}")
    
    # Save checkpoint after each epoch
    epoch_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
    os.makedirs(epoch_checkpoint_dir, exist_ok=True)
    
    print(f"Saving checkpoint for epoch {epoch+1} to {epoch_checkpoint_dir}")
    if hasattr(model, 'module'):  # DataParallel case
        model.module.save_pretrained(epoch_checkpoint_dir)
    else:  # Single GPU case
        model.save_pretrained(epoch_checkpoint_dir)
    tokenizer.save_pretrained(epoch_checkpoint_dir)

# Save the final model
print("Model Saved. Training Completed.")

"""
if hasattr(model, 'module'):  # DataParallel case
    model.module.save_pretrained(MODEL_SAVE_PATH)
else:  # Single GPU case
    model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
"""
