# Open the input file
with open('./mutations(en-san).txt', 'r') as file:
    lines = file.readlines()

# Initialize lists to store the extracted sentences
ftrain_src = []
ftrain_tgt = []

# Iterate through the lines in the file
i = 0
while i < len(lines):
    # Check if the line contains 'san_mutated'
    if 'san_mutated:' in lines[i]:
        san_mutated = lines[i].strip().replace('san_mutated:', '').strip()
        # Check if the next line contains 'en_mutated' and ends with '- Correct'
        if i + 2 < len(lines) and 'en_mutated:' in lines[i + 2] and '- Correct' in lines[i + 2]:
            en_mutated = lines[i + 2].strip().replace('en_mutated:', '').replace('- Correct', '').strip()
            # Append to the respective lists
            ftrain_src.append(en_mutated)
            ftrain_tgt.append(san_mutated)
    i += 1

# Write the extracted sentences to the output files
with open('train_mutated.src', 'w') as src_file:
    src_file.write('\n'.join(ftrain_src))

with open('train_mutated.tgt', 'w') as tgt_file:
    tgt_file.write('\n'.join(ftrain_tgt))

print("Extraction complete. Check train_mutated.src and train_mutated.tgt files.")