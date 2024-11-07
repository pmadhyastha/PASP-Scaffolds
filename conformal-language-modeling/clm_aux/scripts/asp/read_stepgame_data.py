# -*- coding: utf-8 -*-

"""
Spyder Editor

"""
import _pickle as pickle
import json

#input_reference_path  = "C:/Users/nkaur/Downloads/valid_curl.json"
#output_name = "C:/Users/nkaur/Downloads/valid_curl_100json.json"

#input_reference_path  = "C:/Users/nkaur/Downloads/valid_random_curl.json"
output_name = "C:/Users/nkaur/Downloads/valid_random_curl_1000json.json"

valid_path       = "C:/Users/nkaur/Downloads/SpatialLM-StepGame-main/SpatialLM-StepGame/valid.json"
valid_label_path = "C:/Users/nkaur/Downloads/SpatialLM-StepGame-main/SpatialLM-StepGame/valid_label.json"


def extract_first1000datapoints(valid_path, valid_label_path):
    valid_data  = json.load(open(valid_path, "rb"))
    valid_label  = pickle.load(open(valid_label_path, "rb")) # valid label is already a dictionary

    myjsonfile = []    
    i = 1            
    for sample1 in valid_data:         
        if  len(sample1['input'].split("\n"))-1 == 1:
            sample1['target'] = valid_label[sample1['input']+sample1['output']]
            myjsonfile.append(sample1) 
            if i >= 1000:
                break
            else:
                i = i +1
    #print(myjsonfile)   
    json_object = json.dumps(myjsonfile, indent=4) 
    
    with open(output_name, "w") as w:  
        w.write(json_object)

def main():
    extract_first1000datapoints(valid_path,valid_label_path )

if __name__ == "__main__":
    main()