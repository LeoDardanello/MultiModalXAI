import csv
import json
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def generate_json_file(file_and_dest):
    for file in file_and_dest: 
        counter = 0
        data = []

        with open(file[0], newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            
            for row in reader:
                counter += 1
                data.append(row)
            print(f"counter: {counter}")

        if not os.path.exists(file[1]):
            with open(file[1], 'w', encoding='utf-8') as jsonfile:
                json.dump([], jsonfile, ensure_ascii=False, indent=4)
            print(f"File JSON vuoto creato come {file[1]}")

            with open(file[1], 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, ensure_ascii=False, indent=4)

            print(f"File JSON salvato come {file[1]}")

def img_plot_html(img_plot): # the parameter figure is of type Figure of Matplotlib
    tmpfile = BytesIO()
    img_plot.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    img_plot_html = "<div style='text-align: center;'><img src=\'data:image/png;base64,{}\'></div>".format(encoded)
    return img_plot_html


def custom_word_tokenizer(s, return_offsets_mapping=True): # need to write a custom function, because shap library works with 'hugging-face' tokenizer style
    tokens = word_tokenize(s)
    input_ids = tokens  # For simplicity, we directly use tokens as input_ids
    
    if return_offsets_mapping:
        # Create offset mapping
        offset_mapping = []
        pos = 0
        for token in tokens:
            start = s.find(token, pos)
            end = start + len(token)
            offset_mapping.append((start, end))
            pos = end
        return {"input_ids": input_ids, "offset_mapping": offset_mapping}
    return {"input_ids": input_ids}
