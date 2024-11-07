import math
import torch
import intel_extension_for_pytorch as ipex
from tqdm.auto import tqdm
from datasets import load_dataset
import os
import sys
import json
import time
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import _pickle as pickle

#from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import (
    HfArgumentParser,
    set_seed,
)
#from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')

stop_sentence = ["###", "Note", "note", "Please", "please", "Example"]

LLAMA_PATH = "<PATH_TO_HUGGINGFACE_MODEL>"
#model_path = "/rds/user/dn-kaur1/hpc-work/LLaMA-Factory/huggingface-models/"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

logger = logging.getLogger(__name__)

#few_shot_asp = """You are an text to Answer Set Program (ASP) translator.Users will ask you questions in English and you will generate a ASP program based on the provided  input.\n Example 1:\n ### Instruction:\n Generate the Answer Set Program(ASP) for the given input problem. Each numbered point in the input will have one corresponding ASP fact as output. Additionally, include some generic ASP rules that will help the user in solving the problem. 1 X is to the left of K and is on the same horizontal plane.\n2 What is the relation of the agent X to the agent K? \n\n### Response:\n % we generate one fact for each input sentence: \n% 1 X is to the left of K and is on the same horizontal plane.\nleft(\"X\", \"K\").\n\n% 2 What is the relation of the agent X to the agent K?\nquery(\"X\", \"K\").\n\n% Now we consider the answer set rules that will help us arrive at the final answer: \n% assume the 2nd queried object is at location (0,0)\nlocation(Q2, 0, 0) :- query(_, Q2).\n\n% extract answer relation R such that the offset (Ox,Oy) of R is in the same direction of (X,Y)\nanswer(R) :- query(Q1, _), location(Q1, X, Y), offset(R, Ox, Oy), Ox=-1: X<0; Ox=0: X=0; Ox=1: X>0; Oy=-1: Y<0; Oy=0: Y=0; Oy=1: Y>0.\n\n% general format translation, which can also be easily done in python script\n% (this is not needed if we directly extract the general form in the beginning as in bAbI task4)\nis(A, top, B) :- top(A, B).\nis(A, top, B) :- up(A, B).\nis(A, down, B) :- down(A, B).\nis(A, left, B) :- left(A, B).\nis(A, right, B) :- right(A, B).\nis(A, top_left, B) :- top_left(A, B).\nis(A, top_right, B) :- top_right(A, B).\nis(A, down_left, B) :- down_left(A, B).\nis(A, down_right, B) :- down_right(A, B).\nis(A, east, B) :- east(A, B).\nis(A, west, B) :- west(A, B).\nis(A, south, B) :- south(A, B).\nis(A, north, B) :- north(A, B).\n\n% synonyms\nsynonyms(north, northOf; south, southOf; west, westOf; east, eastOf; top, northOf; down, southOf; left, westOf; right, eastOf).\nsynonyms(A, B) :- synonyms(B, A).\nsynonyms(A, C) :- synonyms(A, B), synonyms(B, C), A!=C.\n\n% define the offsets of 8 spacial relations\noffset(overlap,0,0; top,0,1; down,0,-1; left,-1,0; right,1,0; top_left,-1,1; top_right,1,1; down_left,-1,-1; down_right,1,-1).\n\n% derive the kind of spacial relation from synonyms and offset\nis(A, R1, B) :- is(A, R2, B), synonyms(R1, R2).\nis(A, R1, B) :- is(B, R2, A), offset(R2,X,Y), offset(R1,-X,-Y).\n\n% derive the location of every object\n% the search space of X or Y coordinate is within -100 and 100 (to avoid infinite loop in clingo when data has error)\nnums(-100..100).\n\nlocation(A, Xa, Ya) :- location(B, Xb, Yb), nums(Xa), nums(Ya), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy.\n\nlocation(B, Xb, Yb) :- location(A, Xa, Ya), nums(Xb), nums(Yb), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy. Example 2:\n ### Instruction:\n Generate the Answer Set Program(ASP) for the given input problem. Each numbered point in the input will have one corresponding ASP fact as output. Additionally, include some generic ASP rules that will help the user in solving the problem. 1 G is at the 6 o'clock position relative to R.\n2 What is the relation of the agent G to the agent R? \n\n### Response:\n % we generate one fact for each input sentence: \n% 1 G is at the 6 o'clock position relative to R.\ndown(\"G\", \"R\").\n\n% 2 What is the relation of the agent G to the agent R?\nquery(\"G\", \"R\").\n\n% Now we consider the answer set rules that will help us arrive at the final answer: \n% assume the 2nd queried object is at location (0,0)\nlocation(Q2, 0, 0) :- query(_, Q2).\n\n% extract answer relation R such that the offset (Ox,Oy) of R is in the same direction of (X,Y)\nanswer(R) :- query(Q1, _), location(Q1, X, Y), offset(R, Ox, Oy), Ox=-1: X<0; Ox=0: X=0; Ox=1: X>0; Oy=-1: Y<0; Oy=0: Y=0; Oy=1: Y>0.\n\n% general format translation, which can also be easily done in python script\n% (this is not needed if we directly extract the general form in the beginning as in bAbI task4)\nis(A, top, B) :- top(A, B).\nis(A, top, B) :- up(A, B).\nis(A, down, B) :- down(A, B).\nis(A, left, B) :- left(A, B).\nis(A, right, B) :- right(A, B).\nis(A, top_left, B) :- top_left(A, B).\nis(A, top_right, B) :- top_right(A, B).\nis(A, down_left, B) :- down_left(A, B).\nis(A, down_right, B) :- down_right(A, B).\nis(A, east, B) :- east(A, B).\nis(A, west, B) :- west(A, B).\nis(A, south, B) :- south(A, B).\nis(A, north, B) :- north(A, B).\n\n% synonyms\nsynonyms(north, northOf; south, southOf; west, westOf; east, eastOf; top, northOf; down, southOf; left, westOf; right, eastOf).\nsynonyms(A, B) :- synonyms(B, A).\nsynonyms(A, C) :- synonyms(A, B), synonyms(B, C), A!=C.\n\n% define the offsets of 8 spacial relations\noffset(overlap,0,0; top,0,1; down,0,-1; left,-1,0; right,1,0; top_left,-1,1; top_right,1,1; down_left,-1,-1; down_right,1,-1).\n\n% derive the kind of spacial relation from synonyms and offset\nis(A, R1, B) :- is(A, R2, B), synonyms(R1, R2).\nis(A, R1, B) :- is(B, R2, A), offset(R2,X,Y), offset(R1,-X,-Y).\n\n% derive the location of every object\n% the search space of X or Y coordinate is within -100 and 100 (to avoid infinite loop in clingo when data has error)\nnums(-100..100).\n\nlocation(A, Xa, Ya) :- location(B, Xb, Yb), nums(Xa), nums(Ya), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy.\n\nlocation(B, Xb, Yb) :- location(A, Xa, Ya), nums(Xb), nums(Yb), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy. \n Example 3:\n ### Instruction:\n"""
few_shot_asp = """You are an text to Answer Set Program (ASP) translator.Users will ask you questions in English and you will generate a ASP program based on the provided  input.\n Example 1:\n ### Instruction:\n Generate the Answer Set Program(ASP) for the given input problem. Each numbered point in the input will have one corresponding ASP fact as output. Additionally, include some generic ASP rules that will help the user in solving the problem. 1 F and T are next to each other with F on the left and T on the right.\n2 U and F are parallel, and U is on top of F.\n3 What is the relation of the agent U to the agent F? \n\n### Response:\n % we generate one fact for each input sentence: \n% 1 F and T are next to each other with F on the left and T on the right.\nright(\"T\", \"F\").\n\n% 2 U and F are parallel, and U is on top of F.\ntop(\"U\", \"F\").\n\n% 3 What is the relation of the agent U to the agent F?\nquery(\"U\", \"F\").\n\n% Now we consider the answer set rules that will help us arrive at the final answer: \n% assume the 2nd queried object is at location (0,0)\nlocation(Q2, 0, 0) :- query(_, Q2).\n\n% extract answer relation R such that the offset (Ox,Oy) of R is in the same direction of (X,Y)\nanswer(R) :- query(Q1, _), location(Q1, X, Y), offset(R, Ox, Oy), Ox=-1: X<0; Ox=0: X=0; Ox=1: X>0; Oy=-1: Y<0; Oy=0: Y=0; Oy=1: Y>0.\n\n% general format translation, which can also be easily done in python script\n% (this is not needed if we directly extract the general form in the beginning as in bAbI task4)\nis(A, top, B) :- top(A, B).\nis(A, top, B) :- up(A, B).\nis(A, down, B) :- down(A, B).\nis(A, left, B) :- left(A, B).\nis(A, right, B) :- right(A, B).\nis(A, top_left, B) :- top_left(A, B).\nis(A, top_right, B) :- top_right(A, B).\nis(A, down_left, B) :- down_left(A, B).\nis(A, down_right, B) :- down_right(A, B).\nis(A, east, B) :- east(A, B).\nis(A, west, B) :- west(A, B).\nis(A, south, B) :- south(A, B).\nis(A, north, B) :- north(A, B).\n\n% synonyms\nsynonyms(north, northOf; south, southOf; west, westOf; east, eastOf; top, northOf; down, southOf; left, westOf; right, eastOf).\nsynonyms(A, B) :- synonyms(B, A).\nsynonyms(A, C) :- synonyms(A, B), synonyms(B, C), A!=C.\n\n% define the offsets of 8 spacial relations\noffset(overlap,0,0; top,0,1; down,0,-1; left,-1,0; right,1,0; top_left,-1,1; top_right,1,1; down_left,-1,-1; down_right,1,-1).\n\n% derive the kind of spacial relation from synonyms and offset\nis(A, R1, B) :- is(A, R2, B), synonyms(R1, R2).\nis(A, R1, B) :- is(B, R2, A), offset(R2,X,Y), offset(R1,-X,-Y).\n\n% derive the location of every object\n% the search space of X or Y coordinate is within -100 and 100 (to avoid infinite loop in clingo when data has error)\nnums(-100..100).\n\nlocation(A, Xa, Ya) :- location(B, Xb, Yb), nums(Xa), nums(Ya), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy.\n\nlocation(B, Xb, Yb) :- location(A, Xa, Ya), nums(Xb), nums(Yb), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy. Example 2:\n ### Instruction:\n Generate the Answer Set Program(ASP) for the given input problem. Each numbered point in the input will have one corresponding ASP fact as output. Additionally, include some generic ASP rules that will help the user in solving the problem. 1 C and M are both there with the object C above the object M.\n2 Z is at the bottom and Y is on the top.\n3 Z is at a 45 degree angle to M, in the upper lefthand corner.\n4 Y is placed at the lower left of G.\n5 What is the relation of the agent Z to the agent C? \n\n### Response:\n % we generate one fact for each input sentence: \n% 1 C and M are both there with the object C above the object M.\ntop(\"C\", \"M\").\n\n% 2 Z is at the bottom and Y is on the top.\ndown(\"Z\", \"Y\").\n\n% 3 Z is at a 45 degree angle to M, in the upper lefthand corner.\ntop_left(\"Z\", \"M\").\n\n% 4 Y is placed at the lower left of G.\ndown_left(\"Y\", \"G\").\n\n% 5 What is the relation of the agent Z to the agent C?\nquery(\"Z\", \"C\").\n\n% Now we consider the answer set rules that will help us arrive at the final answer: \n% assume the 2nd queried object is at location (0,0)\nlocation(Q2, 0, 0) :- query(_, Q2).\n\n% extract answer relation R such that the offset (Ox,Oy) of R is in the same direction of (X,Y)\nanswer(R) :- query(Q1, _), location(Q1, X, Y), offset(R, Ox, Oy), Ox=-1: X<0; Ox=0: X=0; Ox=1: X>0; Oy=-1: Y<0; Oy=0: Y=0; Oy=1: Y>0.\n\n% general format translation, which can also be easily done in python script\n% (this is not needed if we directly extract the general form in the beginning as in bAbI task4)\nis(A, top, B) :- top(A, B).\nis(A, top, B) :- up(A, B).\nis(A, down, B) :- down(A, B).\nis(A, left, B) :- left(A, B).\nis(A, right, B) :- right(A, B).\nis(A, top_left, B) :- top_left(A, B).\nis(A, top_right, B) :- top_right(A, B).\nis(A, down_left, B) :- down_left(A, B).\nis(A, down_right, B) :- down_right(A, B).\nis(A, east, B) :- east(A, B).\nis(A, west, B) :- west(A, B).\nis(A, south, B) :- south(A, B).\nis(A, north, B) :- north(A, B).\n\n% synonyms\nsynonyms(north, northOf; south, southOf; west, westOf; east, eastOf; top, northOf; down, southOf; left, westOf; right, eastOf).\nsynonyms(A, B) :- synonyms(B, A).\nsynonyms(A, C) :- synonyms(A, B), synonyms(B, C), A!=C.\n\n% define the offsets of 8 spacial relations\noffset(overlap,0,0; top,0,1; down,0,-1; left,-1,0; right,1,0; top_left,-1,1; top_right,1,1; down_left,-1,-1; down_right,1,-1).\n\n% derive the kind of spacial relation from synonyms and offset\nis(A, R1, B) :- is(A, R2, B), synonyms(R1, R2).\nis(A, R1, B) :- is(B, R2, A), offset(R2,X,Y), offset(R1,-X,-Y).\n\n% derive the location of every object\n% the search space of X or Y coordinate is within -100 and 100 (to avoid infinite loop in clingo when data has error)\nnums(-100..100).\n\nlocation(A, Xa, Ya) :- location(B, Xb, Yb), nums(Xa), nums(Ya), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy.\n\nlocation(B, Xb, Yb) :- location(A, Xa, Ya), nums(Xb), nums(Yb), is(A, Kind, B), offset(Kind, Dx, Dy), Xa-Xb=Dx, Ya-Yb=Dy. \n Example 3:\n ### Instruction:\n"""


@dataclass
class PredictArguments:
    """
    Arguments pertaining to how to run the prediction.
    """

    input_name: str = field(default='valid_curl_length1-5_1000json.json')
    output_name: str = field(default='dev_metallama3-8b-instruct-100_examples_part2.jsonl')
    
    checkpoint: str = field(default=LLAMA_PATH)
    strategy: str = field(default="sample")
    predict_split: str = field(default="validation")
    num_generations: int = field(default=20)
    max_predict_samples: Optional[int] = field(default=None)
    #max_predict_samples: int = field(default=100)
    starting_x: int = field(default=0)
    seed: int = field(default=42)
    few_shot: int = field(default=32)
    batch_size: int = field(default=4)
    stepgame_starting_index: int = field(default=0)
    stepgame_end_index: int = field(default=0)


def main():
    parser = HfArgumentParser(PredictArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()

    args: PredictArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Predicting with {args}")

    logger.info("Loading dataset")

    with open(args.input_name) as f:
        pickle_dataset = json.load(f)

    prompt  = few_shot_asp

    #logger.info("Using prompt: %s", prompt)

    if args.max_predict_samples is not None:
       dataset = pickle_dataset[args.starting_x:args.starting_x + args.max_predict_samples]
    else:
       dataset = pickle_dataset

    logger.info("Loading model")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint,low_cpu_mem_usage=True, torch_dtype=torch.float32)

    datatype=""
    for name, param in model.named_parameters():
       datatype = param.dtype
       break

    logger.info("Moving model to device")
    logger.info(datatype)

    model = model.bfloat16()
    model = model.to('xpu')
    torch.autocast(device_type="xpu", enabled=True, dtype=datatype)

    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id) # <|eot_id|>
    tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id) # <|begin_of_text|>
    #tokenizer.padding_side = "left"
    #logger.info("<|end_of_text|>")
    #logger.info(tokenizer.all_special_tokens)
    #logger.info(tokenizer.decode(128009)) # <|eot_id|>
    #logger.info(tokenizer.decode(128001)) # <|end_of_text|>
    #logger.info(tokenizer.special_tokens_map)
    #logger.info(tokenizer.encode('<|end_of_text|>', return_tensors='pt'))

    def convert_string_to_stopword_ids(stop_sentence, tokenizer):

       stop_word_ids =[]
       for onestopword in stop_sentence:

           stop_word_ids.append(tokenizer.convert_tokens_to_ids(onestopword))
     
       return stop_word_ids

    class StoppingCriteriaSub(transformers.StoppingCriteria):
        def __init__(self, input_length=0, stop_ids=None):
            super().__init__()
            self.stop_ids = stop_ids
            self.input_length = input_length

        def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> bool:
            if self.stop_ids is None:
                return False

            output = input_ids[:, self.input_length:]
            last_token = input_ids[:,-1]
            has_stop_ids = []
            for stop_id in self.stop_ids:
                has_stop_id = torch.any(output == stop_id, dim=1)
                has_stop_ids.append(has_stop_id)
            has_stop_ids = torch.stack(has_stop_ids, dim=1)
            
            return (has_stop_ids.any(dim=1).all())

    stop_word_ids = convert_string_to_stopword_ids(stop_sentence, tokenizer)
    def run(sample, num_generations, example_id):

        question = sample['question']
        answer = sample['answer']
        
        input_text = prompt+ sample['question']
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("xpu")
        #stop_word_ids = convert_string_to_stopword_ids(stop_sentence, tokenizer)
        
        stopping_criteria = transformers.StoppingCriteriaList([StoppingCriteriaSub(stop_ids=stop_word_ids, input_length=input_ids.shape[1])])
        torch.manual_seed(args.seed)

        num_return_sequences = min(num_generations, args.batch_size)
        num_batches = math.ceil(num_generations / num_return_sequences)

        kwargs = {
            "max_new_tokens": 1000,
            "pad_token_id":tokenizer.eos_token_id,           
            "output_scores": True,
            "return_dict_in_generate": True,
            "num_return_sequences": num_return_sequences,
            "stopping_criteria": stopping_criteria,
            #"temperature": 0.6, # remember to turn this on when running the finetuning results
            #"top_p": 0.9
        }

        if args.strategy == "greedy":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
        elif args.strategy == "sample":
            kwargs["do_sample"] = True


        generations = []

        for batch in range(num_batches):
            
            set_seed(args.seed + batch)
            with torch.no_grad():
                start=time.time()
                outputs = model.generate(input_ids, **kwargs)
                timediff=time.time()-start
            logger.info("Example id: {}, time taken for generation: {} time now: {}".format(example_id, timediff, time.time()))    

            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]

            for i in range(len(generated_tokens)):
                tokens = []
                scores = []
                for tok, score in zip(generated_tokens[i], transition_scores[i]):
                    if tok in stop_word_ids and len(tokens) > 0: # avoid the edge case of empty generation
                        break
                    tokens.append(tok)
                    scores.append(score)

                tokens = torch.stack(tokens, dim=0)
                scores = torch.stack(scores, dim=0)

                generations.append({
                    'tokens': tokens.cpu().tolist(),
                    'scores': scores.cpu().tolist(),
                    'decoded': tokenizer.decode(tokens)
                })
        
        #torch.set_printoptions(threshold=sys.maxsize)                 
        #print("generated_tokens: {}".format(generations))
        
        datum = {
            'question': question,
            'answer': answer,
            'generations': generations
        }
        return datum

    output_dir = os.path.dirname(args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    
    i = 0
    j = args.stepgame_starting_index
    k = args.stepgame_end_index
    
    with open(args.output_name, "w") as w:
        for sample1 in tqdm(dataset):
           sample = dict()
           sample['question'] = sample1["instruction"]+" "+sample1["input"]+"\n\n### Response:"     
           sample['answer'] = sample1["output"]
           
           
           if i < j or i > k:
                logger.info("example "+str(i))
                i = i+1
                continue
           
           datum = run(sample, args.num_generations, i)
           w.write(json.dumps(datum) + "\n")
           i = i+1

if __name__ == "__main__":
    main()

