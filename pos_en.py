import stanza
from tqdm import tqdm
import torch

# Download the English model
stanza.download('en')

# Check if GPU is available
use_gpu = torch.cuda.is_available()
print(f"GPU available: {use_gpu}")

# Initialize the pipeline with GPU support if available
nlp = stanza.Pipeline('en', use_gpu=use_gpu)

# Read sentences from the file
input_file = './Dataset/English-Sanskrit/org_data/train2.src'
with open(input_file, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# Initialize dictionaries to store categorized words
pos_categories = {
    'VERB': [],
    'NOUN': [],
    'ADV': [],
    'ADJ': []
}

print("Starting POS tagging...")

# Process each sentence with a progress bar
for sentence in tqdm(sentences, desc="Processing sentences"):
    doc = nlp(sentence)
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in pos_categories:
                pos_categories[word.upos].append(word.text)

print("POS tagging completed. Writing results to file...")

# Write the results to pos.txt
output_file = 'pos_en(en-san).txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for category, words in pos_categories.items():
        file.write(f"{category}->{', '.join(words)}\n")

print(f"Results saved to {output_file}")
