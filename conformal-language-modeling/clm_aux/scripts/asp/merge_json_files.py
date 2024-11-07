# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json

output_path = "<PATH TO INPUT FILE>"
files = "<PATH TO THE FIRST FILE BEING MERGED e.g. metallama3.1-8b-instruct-0_5_examples.jsonl>"
end_point = 200 
def merge_JsonFiles(input_path):
    result = list()
    #print(result)
    i = 6
    with open(output_path, 'w') as w:        
        with open(input_path, 'r') as infile:
            for line in infile:
                w.write(json.dumps(json.loads(line)) + '\n')
        
        for j in range(6, end_point, 5):
            filename=input_path.replace("0_5_examples.jsonl", "{}_{}_examples.jsonl".format(j, j+4))
            print(filename)
            with open(filename, 'r') as infile:
                for line in infile:
                    w.write(json.dumps(json.loads(line)) + '\n')

    #with open('counseling3.json', 'w') as output_file:
    #    json.dump(result, output_file)

merge_JsonFiles(files)