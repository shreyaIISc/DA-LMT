import random

# Read the source files
with open('./Dataset/English-Mizo/org_data/train2.src', 'r', encoding='utf-8') as f:
    src1 = f.readlines()
with open('./Dataset/English-Mizo/train_80mutated.src', 'r', encoding='utf-8') as f:
    src2 = f.readlines()

# Read the target files
with open('./Dataset/English-Mizo/org_data/train.tgt', 'r', encoding='utf-8') as f:
    tgt1 = f.readlines()
with open('./Dataset/English-Mizo/train_80mutated.tgt', 'r', encoding='utf-8') as f:
    tgt2 = f.readlines()


# Combine the pairs
combined = list(zip(src1 + src2, tgt1 + tgt2))

# Shuffle while maintaining correspondence
random.seed(42)  # for reproducibility
random.shuffle(combined)

# Unzip the pairs
shuffled_src, shuffled_tgt = zip(*combined)

# Write the shuffled files
with open('./Dataset/English-Mizo/train_80merged.src', 'w', encoding='utf-8') as f:
    f.writelines(shuffled_src)
with open('./Dataset/English-Mizo/train_80merged.tgt', 'w', encoding='utf-8') as f:
    f.writelines(shuffled_tgt)

print("Merged and shuffled files created") 
