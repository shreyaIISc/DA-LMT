from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

def print_header(message):
    print("\n" + "="*60)
    print(f" {message}")
    print("="*60)

def align_words(english_words, sanskrit_words, model, device):
    # Compute embeddings for English and Sanskrit words
    english_embeddings = model.encode(english_words, 
                                    convert_to_tensor=True, 
                                    device=device,
                                    show_progress_bar=False)
    sanskrit_embeddings = model.encode(sanskrit_words, 
                                     convert_to_tensor=True, 
                                     device=device,
                                     show_progress_bar=False)

    # Compute cosine similarity between all pairs
    cosine_scores = util.cos_sim(english_embeddings, sanskrit_embeddings)

    # Align each English word to the most similar Sanskrit word
    alignments = {}
    for i, en_word in enumerate(english_words):
        best_match_idx = np.argmax(cosine_scores[i].cpu().numpy())
        alignments[en_word] = sanskrit_words[best_match_idx]

    return alignments

def build_dictionary(src_file, tgt_file, model, device, max_pairs=1000000):
    dictionary = defaultdict(list)
    processed_pairs = 0

    with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()

        print(f"\nProcessing {min(len(src_lines), max_pairs)} sentence pairs...")
        
        # Process each sentence pair with progress bar
        for src, tgt in tqdm(zip(src_lines[:max_pairs], tgt_lines[:max_pairs]), 
                           total=min(len(src_lines), max_pairs),
                           desc="Aligning sentences",
                           unit="pair"):
            try:
                src_words = src.strip().split()
                tgt_words = tgt.strip().split()

                if not src_words or not tgt_words:
                    continue

                # Align words using the pre-trained model
                alignments = align_words(tgt_words, src_words, model, device)

                # Add aligned word pairs to the dictionary
                for en_word, san_word in alignments.items():
                    dictionary[en_word].append(san_word)

                processed_pairs += 1
            except Exception as e:
                print(f"\n⚠ Error processing pair {processed_pairs}: {str(e)}")
                continue

    # Remove duplicates in the lists
    print("\nRemoving duplicates...")
    for key in tqdm(dictionary, desc="Deduplicating", unit="word"):
        dictionary[key] = list(set(dictionary[key]))

    return dictionary, processed_pairs

def main():
    print_header("English-Sanskrit Dictionary Builder")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚙ Using device: {device}")
    if str(device) == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load the model
    print("\nLoading multilingual sentence transformer model...")
    try:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")
        return

    # File paths
    src_file = './Dataset/English-Sanskrit/org_data/train.tgt'
    tgt_file = './Dataset/English-Sanskrit/org_data/train2.src'

    # Build the dictionary
    dictionary, processed_count = build_dictionary(src_file, tgt_file, model, device, max_pairs=1000)
    
    # Save the dictionary
    output_file = 'dictionary(en-san).txt'
    print(f"\nSaving dictionary to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for key in tqdm(sorted(dictionary.keys()), desc="Writing dictionary", unit="entry"):
                values = dictionary[key]
                f.write(f"{key}->{', '.join(values)}\n")
        
        # Print summary
        print_header("Summary Statistics")
        print(f"Processed pairs: {processed_count}")
        print(f"Dictionary entries: {len(dictionary)}")
        avg_translations = sum(len(v) for v in dictionary.values())/len(dictionary) if dictionary else 0
        print(f"Average translations per word: {avg_translations:.2f}")
        print(f"\n✓ Dictionary successfully saved to {output_file}")
    except Exception as e:
        print(f"⚠ Failed to save dictionary: {e}")

if __name__ == "__main__":
    main()