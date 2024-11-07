import os
import re
import json
import string
import numpy as np
import argparse
import sys
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

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, answers):
    """True if prediction matches any answer."""
    prediction = re.split(r'[.,\n]', prediction, maxsplit=1)[0]
    prediction = normalize_text(prediction)
    answers = [normalize_text(a) for a in answers]
    return float(any([prediction == a for a in answers]))

def compute_rouge_score(prediction, answer):
    #scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #return scorer.score(prediction, answer)['rougeL'].fmeasure
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(prediction, answer)['rouge1'].fmeasure

def normalized_likelihood(log_probs, alpha=0.6):
    """Likelihood with length penalty."""
    total_log_probs = np.sum(np.clip(log_probs, -1e5, 0))
    penalty = (5 + len(log_probs)) ** alpha / (5 + 1) ** alpha
    return np.exp(total_log_probs / penalty)


def load_data(filename):
    all_losses = []
    all_scores = []
    all_self_eval_scores = []
    with open(os.path.join(filename)) as f:
        for line in tqdm(f):
            example = json.loads(line)
            answers = example['answer']
            predictions = [p['decoded'] for p in example['generations']]
            #self_eval_scores = [p['self_eval'] for p in example['generations']]
            scores = [normalized_likelihood(p['scores']) for p in example['generations']]
            #losses = [1 - compute_rouge_score(p, answers) for p in predictions]
            all_scores.append(scores)
            #all_losses.append(losses)
            #all_self_eval_scores.append(self_eval_scores)
    #return np.array(all_losses), np.array(all_scores), np.array(all_self_eval_scores)
    #return np.array(all_losses), np.array(all_scores)
    return np.array(all_scores)

if __name__ == "__main__":
    FILENAME = "./dev_metallama3-8b-instruct-100_examples.jsonl"

    parser = argparse.ArgumentParser(description='Compute Trivia QA scores')
    parser.add_argument('--filename', type=str, default=FILENAME)

    parser.add_argument('--scores_filename', type=str, default="<PATH TO all_prob_scores.npy>")
    parser.add_argument('--scores_txt_filename', type=str, default="<PATH TO all_prob_scores.txt>")

    args = parser.parse_args()

    # Load examples.
    #all_losses, all_prob_scores = load_data(args.filename)
    all_prob_scores = load_data(args.filename)
    
    #np.set_printoptions(threshold=sys.maxsize)
    #print("all_losses.shape: {}".format(all_losses.shape))
    #print(all_losses)
    
    #print("all_prob_scores: {}".format(all_prob_scores.shape))
    #print(all_prob_scores)
    
    # Save data
    #np.save(args.self_eval_scores_filename, all_self_eval_scores)
    np.save(args.scores_filename, all_prob_scores)
    #np.save(args.losses_filename, all_losses)
    
    score_txt_file = open(args.scores_txt_filename, "w")
    np.savetxt(score_txt_file, all_prob_scores, delimiter=",", fmt='%f')
    #for row in all_prob_scores:
    #    np.savetxt(score_txt_file, row)
    #score_txt_file.close()    
    
    #losses_txt_file = open(args.losses_txt_filename, "w")
    #np.savetxt(losses_txt_file, all_losses, delimiter=",", fmt='%f')
    
    #for row in all_losses:
    #    np.savetxt(losses_txt_file, row)    
    #losses_txt_file.close()    