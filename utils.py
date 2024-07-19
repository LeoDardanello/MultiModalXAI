import csv
import json
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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
