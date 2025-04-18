import random

# Function to read .conllu file into sentence blocks
def read_conllu_blocks(file_paths):
    blocks = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            current_block = []
            for line in file:
                if line.startswith('# sent_id'):
                    if current_block:
                        blocks.append(current_block)
                        current_block = []
                if line.strip():  # Keep non-empty lines
                    current_block.append(line)
                else:
                    if current_block:  # End of a sentence block
                        blocks.append(current_block)
                        current_block = []
            if current_block:  # Edge case: last sentence block
                blocks.append(current_block)
    return blocks

# Function to write blocks back to a .conllu file
def write_conllu_blocks(file_path, blocks):
    with open(file_path, 'w', encoding='utf-8') as file:
        for block in blocks:
            for line in block:
                file.write(line)
            file.write('\n')  # Ensure spacing between blocks

# Input files
input_files = [
    'examples/pa-ud-dev.conllu',
    'examples/pa-ud-test.conllu',
    'examples/pa-ud-train.conllu'
]

# Output file
output_file = 'examples/pa-combined-data.conllu'

# Read and shuffle blocks
blocks = read_conllu_blocks(input_files)
random.shuffle(blocks)  # Shuffle sentence blocks, not lines

# Write combined shuffled file
write_conllu_blocks(output_file, blocks)
print(f'Combined shuffled file created: {output_file}')

# Split into 5 equal folds
num_folds = 5
folds = [[] for _ in range(num_folds)]

# Distribute blocks evenly across folds
for i, block in enumerate(blocks):
    folds[i % num_folds].append(block)

# Save each fold separately and print fold sizes
fold_paths = []
for i, fold in enumerate(folds):
    fold_path = f'folds/initial/pa-fold-{i + 1}.conllu'
    write_conllu_blocks(fold_path, fold)
    fold_paths.append(fold_path)
    print(f'Fold {i + 1} saved: {fold_path} | Size: {len(fold)} blocks')

# Confirm fold sizes are balanced
fold_sizes = [len(fold) for fold in folds]
print(f'\nFold Sizes: {fold_sizes}')
