import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu import corpus_bleu
from tqdm import tqdm

# Configuration
MODEL_CHECKPOINT = "./re-checkpoint-hindi/epoch_5"  # Path to your trained model
SHUFFLED_FILE = "./Dataset/English-Hindi/org_data/shuffled_hindi.tgt"               # Path to shuffled sentences
TARGET_FILE = "./Dataset/English-Hindi/org_data/dev-1000.tgt"                    # Path to correct sentences
PRED_FILE = "rearranged-hi.txt"                       # Output for predictions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)
model.eval()

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# Load data
shuffled_sentences = load_sentences(SHUFFLED_FILE)
target_sentences = load_sentences(TARGET_FILE)

assert len(shuffled_sentences) == len(target_sentences), "Mismatch in number of sentences"

# Make predictions
predictions = []
for sent in tqdm(shuffled_sentences, desc="Predicting"):
    inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)

# Save predictions
with open(PRED_FILE, 'w', encoding='utf-8') as f:
    for pred in predictions:
        f.write(pred + '\n')

"""
# Calculate BLEU score
# Tokenize references and predictions for BLEU calculation
references = [[ref.split()] for ref in target_sentences]
hypotheses = [pred.split() for pred in predictions]
"""

# Calculate corpus BLEU
bleu_score = corpus_bleu(predictions, [target_sentences])
print(f"Corpus BLEU score: {bleu_score.score:.4f}")

"""
# Calculate individual BLEU scores
individual_bleus = []
for ref, hyp in zip(references, hypotheses):
    individual_bleus.append(sentence_bleu(ref, hyp))

print(f"Average sentence BLEU: {sum(individual_bleus)/len(individual_bleus):.4f}")
"""
