import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from sacrebleu import corpus_bleu
import nltk
from tqdm import tqdm
import time

nltk.download('punkt')

# Configuration
checkpoint_path = "/data1/home/kshreya/Genie2024/GENIE/en-as-nllb/final_model"  # Replace with your checkpoint path
test_src_file = "/data1/home/kshreya/Genie2024/GENIE/Dataset/English-Assamese/org_data/test.src"                   # English sentences to translate
test_tgt_file = "/data1/home/kshreya/Genie2024/GENIE/Dataset/English-Assamese/org_data/test.tgt"                   # Reference Assamese translations
output_file = "en-as-nllb2.txt"                   # Where to save predictions
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # Increased batch size for better GPU utilization

# Print device information
print(f"\n{'='*50}")
print(f"Initializing translation pipeline")
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
print(f"{'='*50}\n")

# Load model and tokenizer from checkpoint
print("Loading model and tokenizer...")
start_time = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds\n")

# Set the source and target language codes
src_lang = "eng_Latn"
tgt_lang = "asm_Beng"

def load_test_data(src_file, tgt_file):
    """Load test data from files"""
    print(f"Loading test data from:\nSource: {src_file}\nTarget: {tgt_file}")
    with open(src_file, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_texts = [line.strip() for line in f]
    
    print(f"Loaded {len(src_texts)} source sentences and {len(tgt_texts)} target references\n")
    return src_texts, tgt_texts

def translate_texts(texts, model, tokenizer, src_lang, tgt_lang, batch_size=16):
    """Translate a list of texts"""
    translations = []
    
    print(f"Starting translation with batch size {batch_size}...")
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
        batch = texts[i:i+batch_size]
        
        # Prepare inputs
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generate translations
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128
        )
        
        # Decode the translations
        batch_translations = tokenizer.batch_decode(
            translated_tokens, 
            skip_special_tokens=True
        )
        
        translations.extend(batch_translations)
    
    total_time = time.time() - start_time
    print(f"\nTranslation completed in {total_time:.2f} seconds")
    print(f"Average speed: {len(texts)/total_time:.2f} sentences/second\n")
    return translations

def main():
    # Load test data
    src_texts, tgt_texts = load_test_data(test_src_file, test_tgt_file)
    
    # Translate the source texts
    translations = translate_texts(src_texts, model, tokenizer, src_lang, tgt_lang, batch_size)
    
    # Save translations to file
    print(f"Saving translations to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')
    
    # Calculate BLEU score
    print("\nCalculating BLEU score...")
    bleu_score = corpus_bleu(translations, [tgt_texts])
    print(f"\n{'='*50}")
    print(f"Final BLEU score: {bleu_score.score:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
