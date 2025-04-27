import transformers
import torch
import json
from tqdm import tqdm
import stanza
import re
import time
import random

random.seed(42)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
model_id = "llama-3.1-8B-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Initialize Stanza pipelines
hi_nlp = stanza.Pipeline('hi', processors='tokenize,pos', use_gpu=torch.cuda.is_available(), dir="/data/shreya/stanza_resources", download_method=None)
en_nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=torch.cuda.is_available(), dir="/data/shreya/stanza_resources", download_method=None)

def tag_hindi_sentence(text, source_id):
    """Tag Hindi sentence with unique POS tags using Stanza"""
    doc = hi_nlp(text)
    tag_counts = {}
    tagged_tokens = []
    word_to_tag = {}
    
    for sentence in doc.sentences:
        for word_idx, word in enumerate(sentence.words):
            pos_tag = word.upos
            tag_counts[pos_tag] = tag_counts.get(pos_tag, 0) + 1
            unique_tag = f"{pos_tag}-{tag_counts[pos_tag]}(s{source_id})"
            tagged_tokens.append(unique_tag)
            word_to_tag[unique_tag] = word.text  # Store mapping from tag to word
    
    return " ".join(tagged_tokens), word_to_tag

def tag_english_sentence(text):
    """Tag English sentence with POS tags using Stanza"""
    doc = en_nlp(text)
    tag_counts = {}
    tagged_tokens = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            pos_tag = word.upos
            tag_counts[pos_tag] = tag_counts.get(pos_tag, 0) + 1
            unique_tag = f"{pos_tag}-{tag_counts[pos_tag]}"
            tagged_tokens.append(unique_tag)
    
    return " ".join(tagged_tokens)

def build_structured_prompt(e1, p1, ph1, e2, p2, ph2):
    """Build a more structured prompt with clear output format requirements"""
    return [
        {"role": "system", "content": """You are an expert linguist performing precise POS tag projection between English and some other language.

Your task is to generate:
1. A new English sentence (e3) combining words from two given sentences
2. Its POS tags (p3) using only tags from the provided examples
3. Projected other langauage POS tags (ph3) based on the observed mappings

FOLLOW THESE RULES STRICTLY:
1. For e3: Combine subset of words from e1 and e2 to form a grammatical sentence
2. For p3: Use ONLY tags from p1 or p2, with source identifiers (s1/s2)
3. For ph3: Use ONLY tags from ph1 or ph2, maintaining the same source identifiers
4. Maintain the same order of tags as words appear in e3
5. The other language POS tags (ph3) must ONLY USE TAGS FROM ph1 or ph2, with source identifiers (s1 or s2). 
6. Learn to map p3 → ph3, using previously observed structural mappings (p1→ph1 and p2→ph2).
7. Other language word order is SOV (Subject-Object-Verb).
8. Drop English DET (the/a) unless other language has an equivalent.
9. Ensure verbs (VERB) agree with the subject's gender/number.
10. Postpositions (ADP) must come after the noun.
11. Pronouns (PRON) must match gender.

INPUT:
e1: English sentences
p1: POS tags with source identifiers of e1
ph1: Other language POS tag with source identifiers 

e2: English sentences
p2: POS tags with source identifiers of e2
ph2: Other language POS tag with source identifiers 

OUTPUT FORMAT:
e3: "[generated english sentence]"
p3: [POS tag sequence]
ph3: [Other language POS tag sequence]
"""},

{"role": "user", "content": f"""

Input Examples:
e1: "{e1}"
p1: {p1}
ph1: {ph1}

e2: "{e2}"
p2: {p2}
ph2: {ph2}

Now STRICTLY ONLY generate e3, p3, and ph3 following all rules and using the format above and DON'T generate long explanation:"""}
    ]

def parse_llm_output(output_text):
    """More robust parsing of LLM output with regex"""
    # Try to find the exact format first
    e3_match = re.search(r'^e3:\s*"(.*?)"\s*$', output_text, re.MULTILINE)
    p3_match = re.search(r'^p3:\s*([A-Za-z0-9\-\(\)\s]+)\s*$', output_text, re.MULTILINE)
    ph3_match = re.search(r'^ph3:\s*([A-Za-z0-9\-\(\)\s]+)\s*$', output_text, re.MULTILINE)
    
    # If exact format not found, try more flexible patterns
    if not all([e3_match, p3_match, ph3_match]):
        e3_match = re.search(r'e3:\s*"(.*?)"', output_text)
        p3_match = re.search(r'p3:\s*([^\n]+)', output_text)
        ph3_match = re.search(r'ph3:\s*([^\n]+)', output_text)
    
    e3 = e3_match.group(1).strip() if e3_match else None
    p3 = p3_match.group(1).strip() if p3_match else None
    ph3 = ph3_match.group(1).strip() if ph3_match else None
    
    if not all([e3, p3, ph3]):
        return None, None, None
    
    # Clean up the tags (remove brackets if present)
    p3 = p3.replace('[', '').replace(']', '').strip()
    ph3 = ph3.replace('[', '').replace(']', '').strip()
    
    # Validate tag formats
    try:
        if not all(re.match(r'^[A-Z]+-\d+(\(s\d\))?$', tag) for tag in p3.split()):
            return None, None, None
        if not all(re.match(r'^[A-Z]+-\d+\(s\d\)$', tag) for tag in ph3.split()):
            return None, None, None
    except:
        return None, None, None
    
    return e3, p3, ph3

def reconstruct_hindi_sentence(ph3, h1_pos_to_word, h2_pos_to_word):
    """Reconstruct Hindi sentence from ph3 using the POS-to-word mappings"""
    words = []
    for tag in ph3.split():
        # Extract source information
        source_id = tag.split('(s')[-1].rstrip(')')
        base_tag = tag.split('(')[0]
        
        if source_id == '1' and tag in h1_pos_to_word:
            words.append(h1_pos_to_word[tag])
        elif source_id == '2'and tag in h2_pos_to_word:
            words.append(h2_pos_to_word[tag])
        else:
            words.append(f'[UNK:{tag}]')
    
    return ' '.join(words)

def generate_with_retry(prompt, max_retries=3):
    """Generate with retries on failure"""
    for _ in range(max_retries):
        try:
            seed = 42
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            output = pipeline(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                # Removed the seed parameter since it's not supported
            )
            return output[0]["generated_text"][-1]["content"]
        except Exception as e:
            print(f"Generation error: {e}")
            continue
    return None

def tag_english_sentence_with_mapping(text, source_id):
    """Tag English sentence with POS tags using Stanza and return mapping"""
    doc = en_nlp(text)
    tag_counts = {}
    tagged_tokens = []
    word_to_tag = {}
    
    for sentence in doc.sentences:
        for word_idx, word in enumerate(sentence.words):
            pos_tag = word.upos
            tag_counts[pos_tag] = tag_counts.get(pos_tag, 0) + 1
            unique_tag = f"{pos_tag}-{tag_counts[pos_tag]}(s{source_id})"
            tagged_tokens.append(unique_tag)
            word_to_tag[word.text.lower()] = unique_tag  # Store mapping from word to tag
    
    return " ".join(tagged_tokens), word_to_tag

def generate_p3_prime(e3, e1_word_map, e2_word_map):
    """Generate p3_prime by looking up words from e3 in the original mappings"""
    p3_prime_tags = []
    words = e3.lower().split()
    
    for word in words:
        # Try to find the word in either mapping (e1 first, then e2)
        tag = e1_word_map.get(word) or e2_word_map.get(word)
        if tag:
            p3_prime_tags.append(tag)
        else:
            # If word not found in either mapping, we can't generate p3_prime
            return None
    
    return " ".join(p3_prime_tags)

def build_ph3_prime_prompt(e1, p1, ph1, e2, p2, ph2, e3, p3_prime):
    """Build a prompt to generate ph3_prime based on p3_prime"""
    return [
        {"role": "system", "content": """You are an expert linguist performing precise POS tag projection between English and some other language.

Your task is to generate projected other language POS tags (ph3_prime) for a new English sentence (e3) based on its POS tags (p3_prime) based on the observed mappings.

FOLLOW THESE RULES STRICTLY:
1. The other language POS tags (ph3_prime) must ONLY USE TAGS FROM ph1 or ph2, with source identifiers (s1 or s2). 
2. Learn to map p3_prime → ph3_prime, using previously observed structural mappings (p1→ph1 and p2→ph2).
3. Other language word order is SOV (Subject-Object-Verb).
4. Drop English DET (the/a) unless other language has an equivalent.
5. Ensure verbs (VERB) agree with the subject's gender/number.
6. Postpositions (ADP) must come after the noun.
7. Pronouns (PRON) must match gender.
  
THINK STEP-BY-STEP:
1. Identify source (s1/s2) for each word in e3.
2. Project POS tags from p3 → ph3 using ph1/ph2.
3. Reorder to other language SOV structure.
4. Validate gender/number agreement.

INPUT:
e1: English sentences
p1: POS tags with source identifiers of e1
ph1: Other language POS tag with source identifiers

e2: English sentences
p2: POS tags with source identifiers of e2
ph2: Other language POS tag with source identifiers

e3: English sentence
p3_prime: English POS tags with source identifiers

OUTPUT FORMAT:
ph3_prime: The corresponding other language POS tags (ph3) ONLY using tags from ph1 or ph2 with source info """},

{"role": "user", "content": f"""

Input Examples:
e1: "{e1}"
p1: {p1}
ph1: {ph1}

e2: "{e2}"
p2: {p2}
ph2: {ph2}

e3: "{e3}"
p3_prime: {p3_prime}

Now STRICTLY ONLY generate ph3_prime following all rules and DON'T generate long explanation:"""}
    ]

def parse_ph3_prime_output(output_text):
    """Parse the ph3_prime from LLM output"""
    match = re.search(r'ph3_prime:\s*([A-Za-z0-9\-\(\)\s]+)(?=\n|$)', output_text)
    if not match:
        return None
    ph3_prime = match.group(1).strip()
    # Validate tag formats
    if not all(re.match(r'^[A-Z]+-\d+\(s\d\)$', tag) for tag in ph3_prime.split()):
        return None
    return ph3_prime



if __name__ == "__main__":
    # Load data
    
    with open('./Dataset/English-Hindi/org_data/train_split.src.4') as f:
        english_sentences = [line.strip() for line in f]
    with open('./Dataset/English-Hindi/org_data/train_split.tgt.4') as f:
        hindi_sentences = [line.strip() for line in f]
    
    """
    with open('./Dataset/en-hi-pseudo/org_data/train.src') as f:
        english_sentences = [line.strip() for line in f]
    with open('./Dataset/en-hi-pseudo/org_data/train.tgt') as f:
        hindi_sentences = [line.strip() for line in f]
    """

    results = []
    num_sentences = len(english_sentences)
    
    # Create pairs - each sentence paired with 1 random other sentences
    pairs = []
    for i in range(num_sentences):
        # Get 1 random indices different from i
        random_indices = random.sample([j for j in range(num_sentences) if j != i], 1)
        for j in random_indices:
            pairs.append((i, j))

    start_time3 = time.time()
    for i, j in tqdm(pairs):
        e1, e2 = english_sentences[i], english_sentences[j]
        h1, h2 = hindi_sentences[i], hindi_sentences[j]
        
        # Get POS tags with mappings
        p1, e1_word_map = tag_english_sentence_with_mapping(e1, 1)
        p2, e2_word_map = tag_english_sentence_with_mapping(e2, 2)
        ph1, h1_word_map = tag_hindi_sentence(h1, 1)
        ph2, h2_word_map = tag_hindi_sentence(h2, 2)
        
        # Generate with LLM
        prompt = build_structured_prompt(e1, p1, ph1, e2, p2, ph2)
        start_time = time.time()
        output_text = generate_with_retry(prompt)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        print("output text: ", output_text)
        
        if not output_text:
            results.append({
                'pair_id': f"{i}-{j}",
                'status': 'generation_failed',
                'e1': e1, 'e2': e2,
                'h1': h1, 'h2': h2,
                'execution_time': execution_time
            })
            continue
        
        e3, p3, ph3 = parse_llm_output(output_text)
        
        if not e3:
            results.append({
                'pair_id': f"{i}-{j}",
                'status': 'parsing_failed',
                'output': output_text,
                'e1': e1, 'e2': e2,
                'execution_time': execution_time
            })
            continue
        
        # Reconstruct Hindi
        h3 = reconstruct_hindi_sentence(ph3, h1_word_map, h2_word_map)
        
        if not h3:
            results.append({
                'pair_id': f"{i}-{j}",
                'status': 'reconstruction_failed',
                'e3': e3,
                'ph3': ph3,
                'h1_word_map': h1_word_map,
                'h2_word_map': h2_word_map,
                'execution_time': execution_time
            })
            continue
        
        # Generate p3_prime
        p3_prime = generate_p3_prime(e3, e1_word_map, e2_word_map)
        
        # Successful generation
        result_entry = {
            'pair_id': f"{i}-{j}",
            'status': 'success',
            'e1': e1, 'e2': e2,
            'h1': h1, 'h2': h2,
            'p1': p1, 'p2': p2,
            'ph1': ph1, 'ph2': ph2,
            'e1_word_map': e1_word_map,
            'e2_word_map': e2_word_map,
            'h1_word_map': h1_word_map,
            'h2_word_map': h2_word_map,
            'e3': e3, 'p3': p3,
            'p3_prime': p3_prime, 
            'ph3': ph3, 'h3': h3,
            'output': output_text,
            'execution_time': execution_time
        }

        # Generate ph3_prime if p3_prime exists
        if p3_prime:
            ph3_prime_prompt = build_ph3_prime_prompt(e1, p1, ph1, e2, p2, ph2, e3, p3_prime)
            start_time2 = time.time()
            ph3_prime_output = generate_with_retry(ph3_prime_prompt)
            end_time2 = time.time()
            execution_time2 = end_time2 - start_time2
            if ph3_prime_output:
                ph3_prime = parse_ph3_prime_output(ph3_prime_output)
                if ph3_prime:
                    # Reconstruct Hindi from ph3_prime
                    h3_prime = reconstruct_hindi_sentence(ph3_prime, h1_word_map, h2_word_map)
                    result_entry.update({
                        'ph3_prime': ph3_prime,
                        'h3_prime': h3_prime,
                        'ph3_prime_output': ph3_prime_output,
                        'execution_time2': execution_time2
                    })
        
        results.append(result_entry)
        
    end_time3 = time.time()
    execution_time3 = end_time3 - start_time3
    print("Total time: ", execution_time3)
    with open('cross_mutation_results_try12_llama3.1_4.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
