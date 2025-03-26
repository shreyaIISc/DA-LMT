import stanza
import torch
from tqdm import tqdm

def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Initializing Sanskrit POS Tagger")
    print(f"Using device: {device}")
    print(f"{'='*50}\n")

    # Download the Sanskrit model with progress indication
    print("Checking/Downloading Sanskrit model...")
    try:
        stanza.download('sa')  # Sanskrit language code
        print("✓ Sanskrit model ready\n")
    except Exception as e:
        print(f"⚠ Error downloading model: {e}")
        return

    # Initialize the pipeline with GPU if available
    print("Initializing NLP pipeline...")
    try:
        nlp = stanza.Pipeline('sa', use_gpu=torch.cuda.is_available())
        print("✓ Pipeline initialized successfully\n")
    except Exception as e:
        print(f"⚠ Pipeline initialization failed: {e}")
        return

    # Read sentences from the file
    input_file = './Dataset/English-Sanskrit/org_data/train.tgt'
    print(f"Reading input file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            sentences = [s.strip() for s in file.readlines() if s.strip()]
        print(f"✓ Loaded {len(sentences)} sentences\n")
    except Exception as e:
        print(f"⚠ Error reading input file: {e}")
        return

    # Initialize dictionaries to store categorized words
    pos_categories = {
        'VERB': [],
        'NOUN': [],
        'ADV': [],
        'ADJ': []
    }

    # Process each sentence with progress bar
    print("Processing sentences...")
    processed_count = 0
    try:
        for sentence in tqdm(sentences, desc="POS Tagging", unit="sent"):
            doc = nlp(sentence)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos in pos_categories:
                        pos_categories[word.upos].append(word.text)
            processed_count += 1
    except Exception as e:
        print(f"\n⚠ Error during processing: {e}")
        print(f"Processed {processed_count}/{len(sentences)} sentences before error")
        return

    # Write the results to pos.txt
    output_file = 'pos_san(en-san).txt'
    print(f"\nWriting results to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for category, words in pos_categories.items():
                file.write(f"{category}->{', '.join(words)}\n")
        
        # Print summary statistics
        print("\nPOS Tagging Summary:")
        print(f"{'-'*30}")
        for category, words in pos_categories.items():
            print(f"{category:<6}: {len(words):>6} words")
        print(f"\n✓ Successfully saved results to {output_file}")
    except Exception as e:
        print(f"⚠ Error writing output file: {e}")

if __name__ == "__main__":
    main()