from flair.models import SequenceTagger
from flair.data import Sentence
import re

# Load the tagger
model = SequenceTagger.load('/raid/home/kshreya/data_aug_nmt/AsPOS.pt')

# Initialize dictionaries to store words by category
pos_categories = {
    'noun': [],
    'verb': [],
    'adjective': [],
    'adverb': []
}

# Read sentences from train.tgt file
with open('./Dataset/English-Assamese/org_data/train.tgt', 'r', encoding='utf-8') as f:
    sentences = f.readlines()

# Process each sentence
for sen in sentences:
    # Clean the sentence if needed
    sen = sen.strip()
    if not sen:
        continue
        
    # Create sentence object and predict tags
    sentence = Sentence(sen)
    model.predict(sentence)
    
    # Iterate through tokens and collect words by POS
    for token in sentence:
        pos_tag = token.labels[0].value
        
        # Map POS tags to our categories (you may need to adjust this based on your tagset)
        if pos_tag.startswith('N'):  # Noun
            pos_categories['noun'].append(token.text)
        elif pos_tag.startswith('V'):  # Verb
            pos_categories['verb'].append(token.text)
        elif pos_tag.startswith('ADJ'):  # Adjective
            pos_categories['adjective'].append(token.text)
        elif pos_tag.startswith('ADV'):  # Adverb
            pos_categories['adjective'].append(token.text)

# Remove duplicates from each category
for category in pos_categories:
    pos_categories[category] = list(set(pos_categories[category]))

# Write the output
output_file = 'pos_as(en-as).txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for category, words in pos_categories.items():
        f.write(f"{category}->{', '.join(words)}\n")

print(f"Results saved to {output_file}")