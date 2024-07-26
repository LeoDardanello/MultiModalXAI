import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class MultimodalDataset(Dataset): # Dataset for handling multimodal data
    def __init__(self, images_dir, json_file_path): # dir_path -> directory path where images are stored / json_file_path -> file path for metadata (including labels)   
        file_paths, text_list, labels_misogyny, shaming_label_list, stereotype_label_list, objectification_label_list, violence_label_list = load_json_file(json_file_path)
   
        self.file_paths = file_paths
        self.images_dir = images_dir
        self.text_list = text_list
        self.labels_misogyny = labels_misogyny
        self.shaming_label_list = shaming_label_list
        self.stereotype_label_list = stereotype_label_list
        self.objectification_label_list = objectification_label_list
        self.violence_label_list = violence_label_list
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        return self.file_paths[idx], self.text_list[idx], self.labels_misogyny[idx], self.shaming_label_list[idx], self.stereotype_label_list[idx], self.objectification_label_list[idx], self.violence_label_list[idx]
    
def load_json_file(json_file_path):
    with open(json_file_path,"r", encoding='utf-8') as f:
        data = json.load(f)
        text_list = [] # list of the text related to each image
        image_list = [] # list of images path
        labels_misogyny = [] # list of TASK A labels (misogyny classification)

        ### list of TASK B labels ###
        shaming_label_list = [] 
        stereotype_label_list = []
        objectification_label_list = []
        violence_label_list = []


        for item in tqdm(data): 
            image_list.append(item['file_name'])
            text_list.append(item["text"])
            labels_misogyny.append(float(item["label"]))
            shaming_label_list.append(float(item["shaming"]))
            stereotype_label_list.append(float(item["stereotype"]))
            objectification_label_list.append(float(item["objectification"]))
            violence_label_list.append(float(item["violence"]))

        return image_list, text_list, torch.tensor(labels_misogyny, dtype=torch.float32), torch.tensor(shaming_label_list, dtype=torch.float32), torch.tensor(stereotype_label_list, dtype=torch.float32), torch.tensor(objectification_label_list, dtype=torch.float32), torch.tensor(violence_label_list, dtype=torch.float32)
    
def accuracy(preds, labels, thresh):
    num_samples = labels.shape[0]
    preds = preds > thresh
    matching_rows = torch.eq(labels.bool(), preds)
    
    # in case we're dealing with the prediction of task B/task A (they've different number of dimensions)
    num_correct = matching_rows.all(dim=1).sum().item() if preds.ndim!=1 else matching_rows.sum().item()
    return 100*(num_correct/num_samples)