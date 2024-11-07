import os
import re
import json
import string
import numpy as np
import argparse
from tqdm.auto import tqdm
from rouge_score import rouge_scorer

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    #def remove_comments_from_asp_program(text):
    #    split = text.split("\n")
        

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def remove_comments_from_asp_program(text):
    split = text.split("\n")
    no_comments = [p for p in split if '%' not in p] 
    #print(no_comments)
    no_spaces = [p for p in no_comments if p != " " and p!=""] 
    #print(no_spaces)    
    no_whitespaces = "\n".join([p.replace(" ", "") for p in no_spaces]) 
    
    return no_whitespaces    

def compute_rouge_score(prediction, answer):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(prediction, answer)['rougeL'].fmeasure
    #scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    #return scorer.score(prediction, answer)['rouge1'].fmeasure

def compute_diversity(filename):
    all_predictions = []
    with open(os.path.join(filename)) as f:
        for line in tqdm(f):
            example = json.loads(line)
            predictions = [remove_comments_from_asp_program(p['decoded']) for p in example['generations']]
            #predictions = [p['decoded'] for p in example['generations']]
            all_predictions.append(predictions)

    N = len(all_predictions)
    g = len(all_predictions[0])

    def compute_match(i):
        arr = np.zeros((g, g))
        for j in range(g):
            for k in range(j, g):
                #arr[j, k] = (all_predictions[i][j]) == (all_predictions[i][k])
                arr[j, k] = compute_rouge_score(all_predictions[i][j], all_predictions[i][k])               
        return arr

    diversity = np.array([compute_match(i) for i in tqdm(range(N))])

    return diversity

if __name__ == "__main__":
    FILENAME = "./dev_metallama3-8b-instruct-100_examples.jsonl"

    parser = argparse.ArgumentParser(description='Compute Trivia QA scores')
    parser.add_argument('--filename', type=str, default=FILENAME)
    parser.add_argument('--output_filename',     type=str, default="<PATH TO diversity.npy>")
    parser.add_argument('--output_txt_filename', type=str, default="<PATH TO diversity.txt>")

    args = parser.parse_args()

    # Load examples.
    diversity = compute_diversity(args.filename)

    # Save data
    np.save(args.output_filename, diversity)
    
    #output_txt_file = open(args.output_txt_filename, "w")
    #np.savetxt(output_txt_file, diversity, delimiter=",", fmt='%f')
    
    with open(args.output_txt_filename, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(diversity.shape))
        
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in diversity:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
