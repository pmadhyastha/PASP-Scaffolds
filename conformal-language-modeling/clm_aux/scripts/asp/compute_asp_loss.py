# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:12:44 2024

@author: nkaur
"""
import json
from tqdm.auto import tqdm
from clingo.control import Control
from clingo.symbol import parse_term
import _pickle as pickle
import numpy as np
import sys
import argparse

class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret

def gen_answer_set(program, opt=False):
        """
        Args:
            program (str): a string of ASP program
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """

        clingo_control = Control(['0', '--warn=none', '--opt-mode=optN', '-t', '4'])
        models = []
        try:
            clingo_control.add('base', [], program)
            clingo_control.ground([('base', [])], context=Context())
        except:
            # breakpoint()
            return []
        if opt:
            clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        else:
            clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        if not models:
            print("Empty List")
        return models

def load_losses(filename):

    accuracy = 0
    i = 1
    labels=[]
    with open(filename, 'r') as f:
        for line in tqdm(f):
            example = json.loads(line)
            
            j = 1
            label_row =[]
            for p in example['generations']:
                answer_sets=gen_answer_set(p["decoded"], opt=False)
                
                if not answer_sets:
                    label_row.append(1) # We want to save loss as loss = 1 - label
                else:
                    label_row.append(0)
                j = j + 1
                
            labels.append(label_row)   
            i = i + 1

    return np.array(labels)

if __name__ == "__main__":
    FILENAME = "<>"

    parser = argparse.ArgumentParser(description='Compute Trivia QA scores')
    parser.add_argument('--filename', type=str, default=FILENAME)
    
    parser.add_argument('--losses_filename', type=str, default=" <PATH TO all_losses.npy>")
    parser.add_argument('--losses_txt_filename', type=str, default="<PATH TO all_losses.txt>")

    args = parser.parse_args()
    
    all_losses = load_losses(args.filename)
    
    np.save(args.losses_filename, all_losses)

    losses_txt_file = open(args.losses_txt_filename, "w")
    np.savetxt(losses_txt_file, all_losses, delimiter=",", fmt='%s')
        