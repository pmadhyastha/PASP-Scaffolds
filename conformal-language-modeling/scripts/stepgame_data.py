"""Split CNN/DM data."""

import argparse
import os
import json
import sys
import numpy as np

INPUT_DIR = ''
OUTPUT_DIR = 'clm_data/stepgame'
INPUT_LOSS_FILE = ''
INPUT_PROB_FILE = ''
INPUT_DIVERSITY_FILE = ''
GENERATION_IDX_FILE =''
SPLITS_FILE = ''
OUTPUT_LABELS_FILE = ''
OUTPUT_PROB_FILE = ''
OUTPUT_DIVERSITY_FILE = ''


def read_jsonl(path, g_shuffle=None):
    """When g_shuffle is not None, permute along axis=1"""

    with open(path) as f:
        raw_data = [json.loads(line) for line in f]

    if g_shuffle is None:
        return raw_data

    data = []
    for i, d in enumerate(raw_data):
        row = []
        for j in range(len(d)):
            row.append(d[g_shuffle[i][j]])
        data.append(row)

    return data

def save_jsonl(dirname, name, data, indices):
    with open(os.path.join(dirname, name), 'w') as f:
        for i in indices:
            f.write(json.dumps(data[i]) + '\n')


def main(args):
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)
    all_losses = np.load(os.path.join(args.input_dir, args.input_loss_file), allow_pickle=True)
    #all_labels = all_losses <= args.loss_threshold
    all_labels = 1 - all_losses
    
    #print("all_losses.shape: {}".format(all_losses.shape)) #all_losses.shape: (no_of_datapoints, no_of_samples)
    #print("all_labels.shape: {}".format(all_labels.shape)) #all_labels.shape: (no_of_datapoints, no_of_samples) 
    #print("all_labels: {}".format(all_labels))
    
    all_probs = np.load(
        os.path.join(args.input_dir, args.input_prob_file), allow_pickle=True)
    all_diversity = np.load(
        os.path.join(args.input_dir, args.input_diversity_file), allow_pickle=True)


    #all_row_rouge = read_jsonl(os.path.join(args.input_dir, 'rouge_scores', 'row_rouge_scores.jsonl'))
    #all_row_reference = read_jsonl(os.path.join(args.input_dir, 'rouge_scores', 'row_reference_idx_to_row_idx.jsonl'))

    # Shuffle examples.
    #e_shuffle = np.random.permutation(len(all_labels))
    e_shuffle = np.arange(len(all_labels))
    
    splits = {
        'train': e_shuffle[:args.num_train],
        'val': e_shuffle[args.num_train:args.num_train + args.num_val],
        'test': e_shuffle[args.num_train + args.num_val:],
    }

    # Shuffle generations.
    g_shuffle = np.ones_like(all_labels).cumsum(axis=-1) - 1
    g_shuffle = g_shuffle.astype('int')
    g_shuffle = np.apply_along_axis(
        np.random.permutation, axis=1, arr=g_shuffle)

    all_labels = np.take_along_axis(all_labels, g_shuffle, axis=1)
    all_probs = np.take_along_axis(all_probs, g_shuffle, axis=1)

    # Permute in both dimensions.
    all_diversity = np.take_along_axis(all_diversity, g_shuffle[:, :, np.newaxis], axis=1)
    all_diversity = np.take_along_axis(all_diversity, g_shuffle[:, np.newaxis, :], axis=2)

    # Read and permute
    #all_row_generation = read_jsonl(os.path.join(args.input_dir, 'rouge_scores', 'row_generation_idx_to_row_idx.jsonl'),  g_shuffle)

    #all_probs_scores = read_jsonl(os.path.join(args.input_dir, 'components', 'probs.jsonl'), g_shuffle)
    #all_nli_scores = read_jsonl(os.path.join(args.input_dir, 'components', 'nli_nocontext.jsonl'), g_shuffle)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, args.generation_idx_file), g_shuffle) # generation_idx_instruction_length_1.npy.shape = (no_of_datapoints, no_of_samples)
    np.savez(os.path.join(args.output_dir, args.splits_file),
             train=splits['train'], val=splits['val'], test=splits['test'])
    mydiversity = []         
    for split, idx in splits.items():
        dirname = os.path.join(args.output_dir, split)
        os.makedirs(dirname, exist_ok=True)
        np.save(os.path.join(dirname, args.output_labels_file), all_labels[idx])
        np.save(os.path.join(dirname, args.output_prob_file), all_probs[idx])
        np.save(os.path.join(dirname, args.output_diversity_file), all_diversity[idx])
        mydiversity = all_diversity[idx]
        #print("all_diversity[idx].shape: {}".format(all_diversity[idx].shape))
    
    #print("all_diversity[idx].shape: {}".format(mydiversity.shape))
    #with open(os.path.join(dirname, 'diversity.txt'), 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        #outfile.write('# Array shape: {0}\n'.format(diversity.shape))
        
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        #for oneslice in mydiversity:
        #for data_slice in diversity:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            #np.savetxt(outfile, oneslice, fmt='%f')

            # Writing out a break to indicate different slices...
            #outfile.write('# New slice\n')    

        #save_jsonl(dirname, 'row_rouge.jsonl', all_row_rouge, idx)
        #save_jsonl(dirname, 'row_generation.jsonl', all_row_generation, idx)
        #save_jsonl(dirname, 'row_reference.jsonl', all_row_reference, idx)
        #save_jsonl(dirname, 'probs_scores.jsonl', all_probs_scores, idx)
        #save_jsonl(dirname, 'nli_scores.jsonl', all_nli_scores, idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--input_loss_file', type=str, default=INPUT_LOSS_FILE)
    parser.add_argument('--input_prob_file', type=str, default=INPUT_PROB_FILE)   
    parser.add_argument('--input_diversity_file', type=str, default=INPUT_DIVERSITY_FILE) 
    parser.add_argument('--generation_idx_file', type=str, default=GENERATION_IDX_FILE)    
    parser.add_argument('--splits_file', type=str, default=SPLITS_FILE)     
    parser.add_argument('--output_labels_file', type=str, default=OUTPUT_LABELS_FILE)  
    parser.add_argument('--output_prob_file', type=str, default=OUTPUT_PROB_FILE) 
    parser.add_argument('--output_diversity_file', type=str, default=OUTPUT_DIVERSITY_FILE)    
    parser.add_argument('--loss_threshold', type=float, default=0.65)
    parser.add_argument('--num_train', type=int, default=60)
    parser.add_argument('--num_val', type=int, default=20)
    args = parser.parse_args()
    main(args)
