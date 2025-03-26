import random
from itertools import combinations, product
import stanza
import torch
from tqdm import tqdm

# Load Stanza with GPU support if available
use_gpu = torch.cuda.is_available()
nlp = stanza.Pipeline(lang="en", processors="tokenize,pos", use_gpu=use_gpu)

print(f"Stanza initialized. Using GPU: {use_gpu}")

def check_word_order(sentence):
    """Check if the sentence follows common English word order using POS tagging."""
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
    # We'll look at sequences of 2-5 tags (covering most basic patterns)
    for length in range(5, 1, -1):
        for i in range(len(pos_tags) - length + 1):
            subsequence = " ".join(pos_tags[i:i+length])
            if subsequence in expected_patterns:
                return "Correct"
    
    return "Incorrect"

# Load POS tags for English and Sanskrit
def load_pos_tags(file_path):
    pos_tags = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '->' in line:
                pos, words = line.strip().split('->')
                pos = pos.strip()
                words = [word.strip() for word in words.split(',')]
                pos_tags[pos] = words
    return pos_tags

# Load dictionary (English -> Sanskrit) with multiple meanings
def load_dictionary(file_path):
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '->' in line:
                en_word, san_meanings = line.strip().split('->')
                en_word = en_word.strip()
                san_meanings = [meaning.strip() for meaning in san_meanings.split(',')]
                if en_word in dictionary:
                    dictionary[en_word].extend(san_meanings)
                else:
                    dictionary[en_word] = san_meanings
    return dictionary

# Load source (Sanskrit) and target (English) sentences
def load_sentences(src_path, tgt_path):
    with open(src_path, 'r', encoding='utf-8') as src_file, open(tgt_path, 'r', encoding='utf-8') as tgt_file:
        src_sentences = [line.strip() for line in src_file]
        tgt_sentences = [line.strip() for line in tgt_file]
    return src_sentences, tgt_sentences

# Function to mutate an English sentence by mutating `k` words at a time
def mutate_sentence_k_words(en_sentence, en_pos_tags, dictionary, k):
    print("---mutate_sentence_k_words---")
    words = en_sentence.split()
    all_replacements = []

    # Generate all combinations of `k` words to mutate
    for indices in combinations(range(len(words)), k):
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
            
            # If the word has a POS tag, replace it with all possible words of the same POS tag
            if pos:
                same_pos_words = en_pos_tags[pos]
                # Filter out the original word and ensure there are valid replacements
                valid_replacements = [w for w in same_pos_words if w != word]
                if valid_replacements:  # Check if there are valid replacements
                    for replacement in valid_replacements:
                        mutated_sentence[i] = replacement
                        replacements[word] = replacement

                        # Join the mutated sentence
                        en_mutated = ' '.join(mutated_sentence)

                        # Check if the mutated sentence makes sense
                        correctness = check_word_order(en_mutated)

                        # Store the mutated sentence along with its correctness status
                        all_replacements.append((en_mutated, replacements.copy(), correctness))
        
    return all_replacements

# Function to update the Sanskrit sentence based on replacements
def update_sanskrit_sentence(san_sentence, replacements, dictionary):
    words = san_sentence.split()
    updated_sentences = []

    # Generate all possible combinations of replacements
    replacement_combinations = []
    for en_word, replacement_word in replacements.items():
        if en_word in dictionary and replacement_word in dictionary:
            e1_meanings = dictionary[en_word]
            e2_meanings = dictionary[replacement_word]
            replacement_combinations.append((e1_meanings, e2_meanings))

    # If no valid replacements, return the original sentence
    if not replacement_combinations:
        return [san_sentence]

    # Generate all possible updated sentences
    for combo in product(*[e2 for e1, e2 in replacement_combinations]):
        updated_sentence = words.copy()
        for (e1_meanings, _), e2_meaning in zip(replacement_combinations, combo):
            for i, word in enumerate(updated_sentence):
                if word in e1_meanings:
                    updated_sentence[i] = e2_meaning
        updated_sentences.append(' '.join(updated_sentence))
    
    return updated_sentences

# Example file paths
pos_train_tgt_path = './pos_en(en-hi).txt'  # English POS tags
pos_train_src_path = './pos_san(en-san).txt'  # Sanskrit POS tags
dictionary_path = './dictionary(en-san).txt'        # English -> Sanskrit dictionary
train_src_path = './Dataset/English-Sanskrit/org_data/train.tgt'              # Sanskrit sentences
train_tgt_path = './Dataset/English-Sanskrit/org_data/train2.src'              # English sentences

# Load POS tags, dictionary, and sentences
print("Loading POS tags and dictionary...")
en_pos_tags = load_pos_tags(pos_train_tgt_path)
san_pos_tags = load_pos_tags(pos_train_src_path)
dictionary = load_dictionary(dictionary_path)

print("Loading source and target sentences...")
src_sentences, tgt_sentences = load_sentences(train_src_path, train_tgt_path)

print("POS tags, dictionary, and sentences loaded successfully.")

# Mutate all sentences and prepare the output
print("Starting mutation process...")
output_lines = []
seen_pairs = set()  # To track unique (en_mutated, san_mutated) pairs

for i, (en_sentence, san_sentence) in enumerate(zip(tgt_sentences, src_sentences)):
    print(f"Processing sentence pair {i + 1}...")
    
    # Generate mutations for 1 word, 2 words, ..., up to all words in the sentence
    for k in range(1, len(en_sentence.split()) + 1):
        # Mutate the English sentence by mutating `k` words at a time
        mutations = mutate_sentence_k_words(en_sentence, en_pos_tags, dictionary, k)
        print("mutations: ", mutations)
        
        for en_mutated, replacements, correctness in mutations:
            # Update the Sanskrit sentence based on replacements
            san_mutations = update_sanskrit_sentence(san_sentence, replacements, dictionary)
            
            # Prepare the output in the desired format
            for san_mutated in san_mutations:
                pair_key = (en_mutated, san_mutated)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    output_lines.append(f"san: {san_sentence}")
                    output_lines.append(f"san_mutated: {san_mutated}")
                    output_lines.append(f"en: {en_sentence}")
                    output_lines.append(f"en_mutated: {en_mutated} - {correctness}")  # Add correctness label
                    output_lines.append("")  # Add a blank line between sentence pairs

print("Mutation process completed.")

# Save the output to mutation.txt
output_file_path = 'mutations(en-san).txt'
print(f"Saving output to {output_file_path}...")

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in output_lines:
        output_file.write(line + '\n')

print(f"Output saved to {output_file_path}.")