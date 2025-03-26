import torch
from tqdm import tqdm
import re

def clean_text(text):
    """
    Convert text to lowercase and remove specified punctuation marks and symbols.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove specified punctuation and symbols
    text = re.sub(r'["\'(),.?!;:@#$%^&*_+=<>{}[\]\\/|`~]', '', text)
    
    return text.strip()

def convert_and_clean(input_file, output_file):
    # Check if GPU is available (though this operation is CPU-bound)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    
    print(f"Processing {len(sentences)} sentences...")
    # Process each sentence with progress bar
    cleaned_sentences = []
    for sentence in tqdm(sentences, desc="Cleaning text"):
        cleaned_sentences.append(clean_text(sentence))
    
    print(f"Writing output to: {output_file}")
    # Write cleaned sentences to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in tqdm(cleaned_sentences, desc="Writing output"):
            file.write(sentence + '\n')

def main():
    input_file = './Dataset/English-Sanskrit/org_data/train.src'  # Path to the input file
    output_file = './Dataset/English-Sanskrit/org_data/train2.src'  # Path to the output file
    
    print("Starting text cleaning process")
    convert_and_clean(input_file, output_file)
    print(f"\nSuccess! Cleaned sentences saved to {output_file}")

if __name__ == "__main__":
    main()