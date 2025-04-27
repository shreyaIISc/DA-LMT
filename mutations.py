import random
from itertools import combinations, product
import stanza
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the multilingual BERT model once at the beginning (outside the function)
# Using 'distiluse-base-multilingual-cased' which is faster and works well for many languages
model = SentenceTransformer('distilbert-base-multilingual-cased')

# Set random seed for reproducibility
random.seed(42)  # You can use any integer value as your seed

# Initialize GPU support
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
print(f"\nInitializing Stanza NLP pipeline... (Using GPU: {use_gpu})")

# Load Stanza with GPU support if available
nlp = stanza.Pipeline(lang="en", processors="tokenize,pos", use_gpu=use_gpu)
print("Stanza pipeline initialized successfully.\n")

def check_word_order(sentence):
    """Check if the sentence follows common English word order using POS tagging."""
    try:
        doc = nlp(sentence)
        
        # Extract POS tags as a list
        pos_tags = [word.upos for sent in doc.sentences for word in sent.words]
        
        # Convert to string for pattern matching
        structure = " ".join(pos_tags)
        
        # Define common English phrase patterns (not just complete sentences)
        expected_patterns = {
            "PRON AUX VERB",        # "She is eating"
            "DET NOUN VERB",        # "The cat sleeps"
            "NOUN VERB DET",        # "Dogs chase the"
            "PRON VERB NOUN",       # "I love books"
            "DET ADJ NOUN",         # "The big dog"
            "PRON VERB",            # "He runs"
            "NOUN VERB",            # "Birds fly"
            "VERB DET NOUN",        # "Eat the apple"
            "AUX VERB NOUN",        # "Is eating lunch"
            "DET NOUN",            # "The house"
            "ADJ NOUN",            # "Big house"
            "VERB ADV",             # "Run quickly"
        }
        
        # Check if any of the expected patterns appear in the sentence structure
        for pattern in expected_patterns:
            if pattern in structure:
                return "Correct"
        
        # For longer sentences, check if we can find consecutive tags that match any pattern
        for length in range(5, 1, -1):
            for i in range(len(pos_tags) - length + 1):
                subsequence = " ".join(pos_tags[i:i+length])
                if subsequence in expected_patterns:
                    return "Correct"
        
        return "Incorrect"
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Error: {str(e)}")
        return "Error"

def load_pos_tags(file_path):
    """Load POS tags from file with progress tracking"""
    print(f"Loading POS tags from {file_path}...")
    pos_tags = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing POS tags"):
            if '->' in line:
                pos, words = line.strip().split('->')
                pos = pos.strip()
                words = [word.strip() for word in words.split(',')]
                pos_tags[pos] = words
    print(f"Loaded {len(pos_tags)} POS categories.\n")
    return pos_tags

def load_dictionary(file_path):
    """Load dictionary with progress tracking"""
    print(f"Loading dictionary from {file_path}...")
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing dictionary"):
            if '->' in line:
                en_word, san_meanings = line.strip().split('->')
                en_word = en_word.strip()
                san_meanings = [meaning.strip() for meaning in san_meanings.split(',')]
                if en_word in dictionary:
                    dictionary[en_word].extend(san_meanings)
                else:
                    dictionary[en_word] = san_meanings
    print(f"Loaded dictionary with {len(dictionary)} entries.\n")
    return dictionary

def load_sentences(src_path, tgt_path):
    """Load sentences with progress tracking"""
    print(f"Loading sentences from {src_path} and {tgt_path}...")
    with open(src_path, 'r', encoding='utf-8') as src_file, \
         open(tgt_path, 'r', encoding='utf-8') as tgt_file:
        src_sentences = [line.strip() for line in tqdm(src_file, desc="Loading source sentences")]
        tgt_sentences = [line.strip() for line in tqdm(tgt_file, desc="Loading target sentences")]
    print(f"Loaded {len(src_sentences)} source and {len(tgt_sentences)} target sentences.\n")
    return src_sentences, tgt_sentences

def mutate_sentence_k_words(en_sentence, en_pos_tags, dictionary, k, num_mutations=5):
    """Mutate k words in a sentence exactly num_mutations times"""
    words = en_sentence.split()
    all_replacements = []
    # Only consider words with length > 3 for word count
    word_count = len([word for word in words if len(word) > 3])
    
    # Skip sentences with less than 2 valid words
    if word_count < 2:
        return []
    
    # Calculate k as exactly 80% of valid word count (rounded to nearest integer)
    k = max(1, round(word_count * 0.8))  # At least 1 word
    
    valid_indices = [i for i, word in enumerate(words) if len(word) > 3 and not word.isdigit()]
    
    if len(valid_indices) < k:
        return []  # Not enough valid words to mutate
    
    # Generate exactly num_mutations mutations
    for _ in range(num_mutations):
        # Randomly select k unique indices to mutate
        indices = random.sample(valid_indices, k)
        mutated_sentence = words.copy()
        replacements = {}

        for i in indices:
            word = words[i]
            # Find the POS tag of the current word
            pos = None
            for tag, word_list in en_pos_tags.items():
                if word in word_list:
                    pos = tag
                    break
            
            # If the word has a POS tag, replace it with a random word of the same POS tag
            if pos:
                same_pos_words = en_pos_tags[pos]
                # Filter out the original word and ensure there are valid replacements
                valid_replacements = [w for w in same_pos_words if w != word and len(w) > 3]
                if valid_replacements:
                    replacement = random.choice(valid_replacements)
                    mutated_sentence[i] = replacement
                    replacements[word] = replacement

        if replacements:  # Only process if we made actual replacements
            # Join the mutated sentence
            en_mutated = ' '.join(mutated_sentence)

            # Check if the mutated sentence makes sense
            correctness = check_word_order(en_mutated)

            # Store the mutated sentence along with its correctness status
            all_replacements.append((en_mutated, replacements.copy(), correctness))
    
    return all_replacements

def update_sanskrit_sentence(san_sentence, replacements, dictionary):
    """Update Sanskrit sentence using BERT semantic similarity"""
    words = san_sentence.split()
    
    for en_word, replacement_word in replacements.items():
        if en_word in dictionary and replacement_word in dictionary:
            e1_meaning = dictionary[en_word][0]  # Original English meaning
            e2_meaning = dictionary[replacement_word][0]  # Replacement meaning
            
            # Encode all words and meaning for similarity comparison
            texts_to_compare = [e1_meaning] + words
            embeddings = model.encode(texts_to_compare, convert_to_tensor=True)
            
            # Calculate similarity between meaning and each word
            meaning_embedding = embeddings[0]
            word_embeddings = embeddings[1:]
            
            # Pairwise cosine similarities
            similarities = torch.nn.functional.cosine_similarity(
                meaning_embedding.unsqueeze(0),
                word_embeddings
            )
            
            # Move tensor to CPU and convert to numpy if needed
            if similarities.is_cuda:
                similarities = similarities.cpu()
            
            # Find best matching word
            best_match_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_match_idx].item()
            
            words[best_match_idx] = e2_meaning
    
    return ' '.join(words)


# Main execution
if __name__ == "__main__":
    # Example file paths
    pos_train_tgt_path = './pos_en-en-arab.txt'  # English POS tags
    #pos_train_src_path = './pos_san(en-mz).txt'  # Sanskrit POS tags
    dictionary_path = './dictionary-en-arab-googleT.txt'  # English -> Sanskrit dictionary
    train_src_path = './Dataset/English-Arabic/org_data/train.tgt'  # Sanskrit sentences
    train_tgt_path = './Dataset/English-Arabic/org_data/train2.src'  # English sentences

    # Load data
    en_pos_tags = load_pos_tags(pos_train_tgt_path)
    #san_pos_tags = load_pos_tags(pos_train_src_path)
    dictionary = load_dictionary(dictionary_path)
    src_sentences, tgt_sentences = load_sentences(train_src_path, train_tgt_path)

    # Mutate all sentences and prepare the output
    print("\nStarting mutation process...")
    output_lines = []
    seen_pairs = set()
    total_pairs = len(src_sentences)
    
    for i, (en_sentence, san_sentence) in enumerate(tqdm(
        zip(tgt_sentences, src_sentences), 
        total=total_pairs, 
        desc="Processing sentence pairs"
    )):
        # Generate exactly 5 mutations per sentence
        mutations = mutate_sentence_k_words(en_sentence, en_pos_tags, dictionary, k=None, num_mutations=5)
        
        for en_mutated, replacements, correctness in mutations:
            san_mutated = update_sanskrit_sentence(san_sentence, replacements, dictionary)
            
            pair_key = (en_mutated, san_mutated)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                output_lines.append(f"san: {san_sentence}")
                output_lines.append(f"san_mutated: {san_mutated}")
                output_lines.append(f"en: {en_sentence}")
                output_lines.append(f"en_mutated: {en_mutated} - {correctness}")
                output_lines.append("")  # Blank line

    print(f"\nGenerated {len(seen_pairs)} unique mutated sentence pairs.")

    # Save the output
    output_file_path = '80wordmutations(en-arab).txt'
    print(f"\nSaving output to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in tqdm(output_lines, desc="Writing output"):
            output_file.write(line + '\n')
    print(f"Successfully saved {len(output_lines)//5} sentence pairs to {output_file_path}")
