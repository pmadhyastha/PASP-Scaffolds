# %%
import numpy as np
import sys
sys.path.append('../')

from clm import utils

# This is the path for the trivia-qa dataset results
#FILENAME = '/rds/user/dn-kaur1/hpc-work/conformal-language-modeling-qa/clm_aux/scripts/qa/clm_samples/test/probs_results.npz'
#OUTPUT_FOLDER = '/rds/user/dn-kaur1/hpc-work/conformal-language-modeling-qa/clm_aux/scripts/qa/clm_samples/samplefig.png'

FILENAME = '/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/clm_results/test/probs_results.npz'
OUTPUT_FOLDER = '/rds/user/dn-kaur1/hpc-work/conformal-language-modeling/clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1/20_samples_per_example/clm_results/test/loss.png'

# This is a path for the stepgame results
#FILENAME = '../scripts/results/triviaqa/test/probs_results.npz'
#OUTPUT_FOLDER = "../scripts/results/triviaqa/test/samplefig.png"
def plot_result(filename):
    output = np.load(FILENAME, allow_pickle=True)
    #print("output['results']")
    #print(output['results'])
    methods, epsilons, results = output['methods'], output['epsilons'], output['results']
    

    utils.print_methods(methods)
    named_results = {
        'First K': results[0],
        #'MaxProb': results[2],
        #'Geometric': results[4],
    }
    ax = utils.plot_results(named_results, epsilons, 'C_relative_excess_avg')
    ax.set_ylabel(r'Relative Excess Samples')
    
    ax = utils.plot_results(named_results, epsilons, 'C_size_avg')
    ax.set_ylabel(r'Set Size')
    
    ax = utils.plot_results(named_results, epsilons, 'C_obj_avg')
    ax.set_ylabel(r'Combined')
    
    ax = utils.plot_results(named_results, epsilons, 'L_avg', add_diagonal=True, ylim_is_xlim=True)
    ax.set_ylabel(r'Set Loss')
    
    ax.figure.savefig(OUTPUT_FOLDER)

# %%
if __name__ == '__main__':
    plot_result(FILENAME)

# %%
"""
FILENAME = '../results/triviaqa/test/probs_results.npz'
plot_result(FILENAME)

# %%
FILENAME = '../results/cnndm/test/probs_results.npz'
plot_result(FILENAME)

# %%
named_results['PlattGeo']['configs'][0]

# %%
is_trivial

# %%
FILENAME = 'results/cxr/val/probs_results.npz'
plot_result(FILENAME)

# %%
FILENAME = 'results/cnndm/val/probs_results.npz'
plot_result(FILENAME)

# %%
"""