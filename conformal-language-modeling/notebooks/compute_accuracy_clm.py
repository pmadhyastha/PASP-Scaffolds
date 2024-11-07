# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import _pickle as pickle
import numpy as np
import sys
import json
import numpy
import argparse
from clingo.control import Control
from clingo.symbol import parse_term

'''
inputindexesfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/clm_results/test/indices_instruction_length_1.pkl" # this file has indices of test data chosen in 100 trials
inputcolumnindexesfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/clm_results/generation_idx_instruction_length_1.npy"
inputclmresultsfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/clm_results/test/probs_results_instruction_length_1.npz" # this file has the conformal sets for 5 methods. The tensor for each method has dimension No_of_trials X No_of_epsilons X No_of_test_examples
samplefilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/metallama3.1-8b-instruct.jsonl" # This file has 20 samples generated for the entire 922 data points
labelfilename = "/home/dn-kaur1/LLaMA-Factory/data/stepgame_valid/valid_curl_1000json.json"
accuracy_path = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/clm_results/test/semantic_accuracy_rougeL_method0.csv"
'''
#inputindexesfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/train_length_1_to_5_test_length_2/20_samples_per_example/clm_results/test/indices_train_length_1_to_5_length_2.pkl"
#inputcolumnindexesfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/train_length_1_to_5_test_length_2/20_samples_per_example/clm_results/generation_idx_train_length_1_to_5_test_length_2.npy"
#inputclmresultsfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/train_length_1_to_5_test_length_2/20_samples_per_example/clm_results/test/probs_train_length_1_to_5_length_2.npz"
#samplefilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/train_length_1_to_5_test_length_2/20_samples_per_example/metallama3.1-8b-instruct.jsonl"
#labelfilename = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/train_length_1_to_5_test_length_2/20_samples_per_example/valid_curl_length1-5_test2_json.json"
#accuracy_path = "/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/train_length_1_to_5_test_length_2/20_samples_per_example/clm_results/test/semantic_accuracy_method0.csv"


INPUTINDEXESFILENAME = ""
INPUTCOLUMNINDEXESFILENAME = ""
INPUTCLMRESULTSFILENAME=""
SAMPLEFILENAME = ""
LABELFILENAME = ""
ACCURACY_PATH = ""

METHOD_ID = 0

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
        return models    

def read_samples(samplefile):
    examples = []
    with open(samplefile) as f:
        for line in f:
            #print(line)
            example = json.loads(line)
            generations = [p['decoded'] for p in example['generations']]
            examples.append(generations)
    
    return examples

def read_labels(labelfilename):
    labels =[]
    with open(labelfilename) as f:
        pickle_dataset = json.load(f)
        for example in pickle_dataset:
            labels.append(example["target"])
    return labels 

def read_trial_indexes(inputfilename):
        indexes = pickle.load(open(inputfilename, "rb"))
        # indexes is a list of 100 elements (100 trials), whose each element is a tuple with 3 elements:
        # each element is a numpy array.
        # indexes[i][0] contains optimization indexes, indexes[i][1] contains calibration indexes, indexes[i][2] contains test indexes for trial i
        
        #print(len(indexes)) # 100 trials
        #print(type(indexes)) # lists
        #print(indexes[0][2]) # test indexes for first trial
        
        return indexes  

def read_column_indexes(inputcolumnfilename):
        generation_indexes = np.load(inputcolumnfilename)
        # generation_indexes is a numpy array of size (total data (oprimization+calibration+test) X no_of_samples)
        
        return generation_indexes 
        

def read_clm_results(inputclmresults):
    output = np.load(inputclmresults, allow_pickle=True)
    methods, epsilons, combined_results = output['methods'], output['epsilons'], output['results']
    #print(methods)
    #print(len(combined_results)) # 5 
    #print(type(combined_results[0])) # dict,  with keys: [L_avg, L_worst_pred_avg, C_size_avg, C_excess_avg,  C_relative_excess_avg, C_obj_avg]
    #print(combined_results[0]['C_indices'].shape) # 1st method, C_indices.shape = no_of_trials X no_of_epsilons X no_of_test_datapoints
    #print(type(combined_results[0]['C_indices'])) # <class 'numpy.ndarray'>
    
    #print(epsilons)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(combined_results[0]['C_indices'])
    #print("read_clm_results")
    #print(combined_results[1]['C_indices'][0][0])
    return combined_results    

def compute_accuracy_per_trial_per_epsilon(per_trial_per_epsilon_prediction_set_size, indices_per_trial, data_per_epsilon, labels_per_epsilon, generation_indexes_per_epsilon):
    
    #print("indices_per_trial: {}".format(indices_per_trial))
    #print("type(indices_per_trial): {}".format(type(indices_per_trial)))
    #print("per_trial_per_epsilon_prediction_set_size: {}".format(per_trial_per_epsilon_prediction_set_size))
    #print("type(per_trial_per_epsilon_prediction_set_size): {}".format(type(per_trial_per_epsilon_prediction_set_size[0]))) 
    
    if np.isnan(per_trial_per_epsilon_prediction_set_size).all():
        return 0
    
    #np.set_printoptions(threshold=sys.maxsize)
    #print("data_per_epsilon: {}".format(data_per_epsilon))  
    #print("type(data_per_epsilon): {}".format(type(data_per_epsilon)))
    #print("labels_per_epsilon.shape: {}".format(labels_per_epsilon.shape)) # labels_per_epsilon.shape = (no_of_test_data_point, )
    
    accuracy = 0
    for i in range(per_trial_per_epsilon_prediction_set_size.shape[0]): # iterate over examples
        #print("index: {}".format(indices_per_trial[i]))
        #print("per_trial_per_epsilon_prediction_set_size[i]: {}".format(per_trial_per_epsilon_prediction_set_size[i]))
        
        #one_data_C_indices = data_per_epsilon[i,0:(int(per_trial_per_epsilon_prediction_set_size[i])+1)]
        one_data_C_indices = data_per_epsilon[i,generation_indexes_per_epsilon[i, 0:(int(per_trial_per_epsilon_prediction_set_size[i])+1)]]
  
        j = 0
        for one_sample in one_data_C_indices: # iterate over samples of one example
            answer_sets=gen_answer_set(one_sample, opt=False)
            j = j +1
                       
            if not answer_sets:
                #print("Case 1. i: {} j: {}".format(i,j))
                continue
                
            if len(answer_sets) != 1:
                #print("**models: {}".format(answer_sets))
                #print("Case 2. i: {} j: {}".format(i,j))
                continue
            if len(answer_sets)== 1 and not answer_sets[0]:    
                #print("**models: {}".format(answer_sets))
                #print("Case 3. i: {} j: {}".format(i,j))
                continue
            
            if len(answer_sets) == 1:
                predicted_answer1 = [s for s in answer_sets[0] if "answer" in s]
                if predicted_answer1:
                    #print("predicted_answer1: {}".format(predicted_answer1))
                    predicted_answer = predicted_answer1[0]
                    if predicted_answer +"."== labels_per_epsilon[i]:
                        accuracy = accuracy + 1
                        #print("Case 4. i: {} j: {}".format(i,j))
                        break
                    #else:
                    #    print("Case 6. i: {} j: {}".format(i,j))
                #else:
                #    print("Case 5. i: {} j: {}".format(i,j))
                    
                    
                        
    print("correct: {} per_trial_per_epsilon_prediction_set_size.shape[0]: {} accuracy: {}".format(accuracy, per_trial_per_epsilon_prediction_set_size.shape[0], (accuracy/per_trial_per_epsilon_prediction_set_size.shape[0])*100))                        
    return (accuracy/per_trial_per_epsilon_prediction_set_size.shape[0])*100        
        
                  
def compute_accuracy(indexes, generation_indexes, results, examples, labels):
    # indexes.shape = no_of_trials X 3 
    #                second dimension 3 is a tuple of  (optimization indexes, calibration indexes, test indexes)
    # results['C_indices'].shape =  no_of_trials X no_of_epsilons X no_of_test_datapoints
    # examples.shape = #test_data X #sample_generations; #test_data = optimization indexes + calibration indexes + test datapoints
      
    #print("results.shape: {}".format(type(results)))   
    
    #np.set_printoptions(threshold=sys.maxsize)
    no_of_trials = results['C_indices'].shape[0]
    no_of_epsilons = results['C_indices'].shape[1]
    
    accuracy_matrix = [ [] for _ in range(no_of_trials)]
    #accuracy_matrix = [ [] for _ in range(2)]
    
    for current_trial in range(no_of_trials):
    #for current_trial in range(2):    
        first_trial_indexes =  indexes[current_trial][2]
        
        #print("generation_indexes")
        #print(generation_indexes)
       
        full_test_data_arr = np.asarray(examples, dtype=object)
        data_per_epsilon = full_test_data_arr[first_trial_indexes]
        full_labels_arr = np.asarray(labels, dtype=object)
        labels_per_epsilon = full_labels_arr[first_trial_indexes]       
        generation_indexes_per_epsilon = generation_indexes[first_trial_indexes]
    
        first_trial_results = results['C_indices'][current_trial]
        
        for current_epsilon in range(no_of_epsilons):
        #for current_epsilon in range(2):
            
            first_epsilon_within_first_trial = first_trial_results[current_epsilon] # class 'numpy.ndarray, first_epsilon_within_first_trial.shape=(no_of_test_data_points,)
    
            #print("first_trial_indexes.shape: {}".format(first_trial_indexes.shape))
            #print("first_trial_results.shape: {}".format(first_trial_results.shape))
            print("first_epsilon_within_first_trial.shape: {}".format(first_epsilon_within_first_trial.shape))
    
            #print("first_trial_indexes: {}".format(first_trial_indexes))   
            #print("first_trial_results: {}".format(first_trial_results))
    
            #print("first_epsilon_within_first_trial: {}".format(first_epsilon_within_first_trial))
                        
            accuracy = compute_accuracy_per_trial_per_epsilon(first_epsilon_within_first_trial, first_trial_indexes, data_per_epsilon, labels_per_epsilon, generation_indexes_per_epsilon)
            print("trial id: {} epsilon id: {}".format(current_trial, current_epsilon))
            accuracy_matrix[current_trial].append(accuracy)
    print("accuracy_matrix")
    print(accuracy_matrix)
            
    print(numpy.array(accuracy_matrix))
    numpy.savetxt(args.accuracy_path, numpy.array(accuracy_matrix), delimiter=",")

def main(args):
    trial_indexes = read_trial_indexes(args.inputindexesfilename)   
    generation_indexes = read_column_indexes(args.inputcolumnindexesfilename)
    combined_results = read_clm_results(args.inputclmresultsfilename)
    examples = read_samples(args.samplefilename)
    labels = read_labels(args.labelfilename)
    
    #print("trial_indexes")
    #print(trial_indexes)
    
    compute_accuracy(trial_indexes, generation_indexes, combined_results[args.method_id], examples, labels)
    
    #combined_results_method_0_indices = combined_results[0]['C_indices']
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_id', type=int, default=METHOD_ID)
    parser.add_argument('--accuracy_path', type=str, default=ACCURACY_PATH)
    parser.add_argument('--inputindexesfilename', type=str, default=INPUTINDEXESFILENAME)
    parser.add_argument('--inputcolumnindexesfilename', type=str, default=INPUTCOLUMNINDEXESFILENAME)
    parser.add_argument('--inputclmresultsfilename', type=str, default=INPUTCLMRESULTSFILENAME)
    parser.add_argument('--samplefilename', type=str, default=SAMPLEFILENAME)
    parser.add_argument('--labelfilename', type=str, default=LABELFILENAME)

    args = parser.parse_args()
    main(args)