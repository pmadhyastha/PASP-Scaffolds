# PASP-Scaffolds
Robust inference with probabilistic answer set program scaffolds for large language models

Sample Generations:

1. cd conformal-language-modeling/clm_aux/scripts/asp/ and run the sample generation file:
   	python run_asp.py --stepgame_starting_index 0 --stepgame_end_index <len(dataset)-1> --batch_size 4 --num_generations 20 --strategy "sample" --output_name './single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct_examples.jsonl'
	
This will generate 20 samples per example and save them in output_name folder
   
2. Run the file code to remove the eos delimiter from the output
   
python asp_fix_eos.py --input "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct_examples.jsonl" --output "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct-examples_fix_eos.jsonl"

3. Compute the loss (admission) function
       
python compute_asp_loss.py --filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct-examples_fix_eos.jsonl" --losses_filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/all_losses_train_length_1_to_5.npy" --losses_txt_filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/all_losses_train_length_1_to_5.txt"

4. compute the quality metric
   
python compute_asp_scores.py --filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct-examples_fix_eos.jsonl" --scores_filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/all_prob_scores_train_length_1_to_5.npy" --scores_txt_filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/all_prob_scores_train_length_1_to_5.txt"

5. compute the diversity metric
   
python compute_asp_intra_match.py --filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct-examples_fix_eos.jsonl" --output_filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/diversity_rouge.npy" --output_txt_filename "./single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/diversity_rouge.txt"

CLM Algorithm run:

6. cd conformal-language-modeling/scripts
   
   Generate the CLM input data
   
   python stepgame_data.py \
               --input_dir "../clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/" \
               --output_dir "../clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/" \
			   --input_loss_file "all_losses_train_length_1_to_5.npy" \
			   --input_prob_file "all_prob_scores_train_length_1_to_5.npy" \
			   --input_diversity_file "diversity_rouge.npy" \
			   --generation_idx_file "generation_idx_instruction_length_1_to_5.npy" \
			   --splits_file "splits_instruction_length_1_to_5.npz" \
			   --output_labels_file "labels_instruction_length_1_to_5.npy" \
			   --output_prob_file "probs_instruction_length_1_to_5.npy" \
			   --output_diversity_file "diversity_instruction_length_1_to_5.npy" \
			   --loss_threshold 0.01 \
			   --num_train 0 \
			   --num_val 0


7. Run the CLM algorithm as follows
   
    python run_trials.py \
                --train_score_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/train/probs_instruction_length_1_to_5.npy" \
                --train_label_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/train/labels_instruction_length_1_to_5.npy" \
           --train_similarity_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/train/diversity_instruction_length_1_to_5.npy" \
                 --test_score_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/probs_instruction_length_1_to_5.npy" \
                 --test_label_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/labels_instruction_length_1_to_5.npy" \
            --test_similarity_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/diversity_instruction_length_1_to_5.npy" \
                     --output_file "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/probs_instruction_length_1_to_5.npz" \
		   --indices_file "indices_instruction_length_1_to_5.pkl"

   8. Run the accuracy code:
       
	cd conformal-language-modeling/notebooks

	python compute_accuracy_clm.py --method_id 0 \
            --accuracy_path "../clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/semantic_accuracy_method0.csv" \
            --inputindexesfilename "../clm_aux/scripts/asp/single-gpu-
                                    generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/indices_instruction_length_1_to_5.pkl" \
            --inputcolumnindexesfilename "../clm_aux/scripts/asp/single-gpu-
                                   generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/generation_idx_instruction_length_1_to_5.npy" \
            --inputclmresultsfilename "../clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/clm_results/test/probs_instruction_length_1_to_5.npz" \
            --samplefilename "../clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/metallama3.1-8b-instruct.jsonl" \
            --labelfilename "../clm_aux/scripts/asp/single-gpu-generations/stepgame/instruction_length_1_to_5/20_samples_per_example/valid_curl_length1-5_test3_json.json"

   
