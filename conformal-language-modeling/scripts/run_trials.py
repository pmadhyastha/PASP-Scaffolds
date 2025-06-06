"""Script to generate outputs."""

import argparse
import os
import numpy as np
import sys
sys.path.append('../')

import _pickle as pickle
from clm import uncertainty
#from clm import uncertainty_transfer
from clm import utils



def main(args):
    utils.set_seed(0)

    # Load dataset.
    train_scores = np.load(args.train_score_file)
    train_labels = np.load(args.train_label_file)
    train_similarity = np.load(args.train_similarity_file)
    train_data = utils.Dataset(train_scores, train_similarity, train_labels)

    test_scores = np.load(args.test_score_file)
    test_labels = np.load(args.test_label_file)
    test_similarity = np.load(args.test_similarity_file)
    test_data = utils.Dataset(test_scores, test_similarity, test_labels)

    #methods, epsilons, results = uncertainty.run_trials(
    #methods, epsilons, results, splits = uncertainty.run_trials(  
    methods, epsilons, results, splits = uncertainty_transfer.run_trials(
        train_data=train_data,
        test_data=test_data,
        p_cal=args.p_cal,
        num_trials=args.num_trials,
        num_processes=args.num_processes)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.savez(
        args.output_file, methods=methods, epsilons=epsilons, results=results)
    
    #print("type(os.path.dirname(args.output_file)): {}".format(type(os.path.dirname(args.output_file))))
    #print("args.indices_file: {}".format(args.indices_file))
    #print("type(args.indices_file): {}".format(type(args.indices_file)))
    output_label = os.path.join(os.path.dirname(args.output_file), args.indices_file)
    print("output_label: {}".format(output_label))
    #output_label=os.path.dirname(args.output_file)+"/"+args.indices_file           #"/indices_instruction_length_1.pkl"
    with open(output_label, 'wb') as file:
        file.write(pickle.dumps(splits)) # use `pickle.loads` to do the reverse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_score_file', type=str)
    parser.add_argument('--train_label_file', type=str)
    parser.add_argument('--train_similarity_file', type=str)
    parser.add_argument('--test_score_file', type=str)
    parser.add_argument('--test_label_file', type=str)
    parser.add_argument('--test_similarity_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--indices_file', type=str)
    parser.add_argument('--p_cal', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--num_processes', type=int, default=40)
    args = parser.parse_args()
    main(args)
