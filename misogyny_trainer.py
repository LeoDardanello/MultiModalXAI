### import torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image # per debug
import os
import json
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from misogyny_dataset import *
from misogyny_classifier import *

class Trainer():
    def __init__(self, train_images_dir,
                       test_image_dir,
                       json_train_path,
                       json_test_path,
                       num_linear_layers=5,
                       drop_value=0.2,
                       train_data_split=0.8,
                       batch_size=256, 
                       lr=0.001, 
                       num_epochs=5,
                       threshold=0.5,
                       weight_taskA=1,
                       weight_taskB=1):
        
        # Check if CUDA is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")
        
        # Loading the Dataset
        self.train_images_dir = train_images_dir
        self.test_image_dir = test_image_dir
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.weight_taskA = weight_taskA
        self.weight_taskB = weight_taskB
        
        train_data = MultimodalDataset(train_images_dir, json_train_path)
        test_data = MultimodalDataset(test_image_dir, json_test_path)
        
        print(f"training on samples:{train_data.__len__()}")
        print(f"testing on samples:{test_data.__len__()}")
    
        self.train_dataloader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
        self.test_dataloader = DataLoader(test_data, batch_size, shuffle=True, pin_memory=True)

        # Defining the Model
        self.classifier = MisogynyCls(num_linear_layers=num_linear_layers, drop_value=drop_value).to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr)
        self.loss_taskA = F.binary_cross_entropy 
        self.loss_taskB = F.binary_cross_entropy

        # Pretrained CLIP loading...
        #self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map='cuda')
        #self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        
    def train_model(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch [{epoch+1}/{self.num_epochs}]')
            train_loss_list, test_loss_list, train_acc_taskA_list, train_acc_taskB_list,  test_acc_taskA_list, test_acc_taskB_list= self.train_epoch()
            
            train_loss_avg = sum(train_loss_list) / len(train_loss_list)
            test_loss_avg = sum(test_loss_list) / len(test_loss_list)
            train_acc_taskA_avg = sum(train_acc_taskA_list) / len(train_acc_taskA_list)
            train_acc_taskB_avg = sum(test_acc_taskA_list) / len(test_acc_taskA_list)
            test_acc_taskA_avg = sum(train_acc_taskB_list) / len(train_acc_taskB_list)
            test_acc_taskB_avg = sum(test_acc_taskB_list) / len(test_acc_taskB_list)
            
            print(f'Average Train Loss: {train_loss_avg: .4f}, Average Test Loss: {test_loss_avg: .4f}')
            print(f'Average Accuracy Train (task A): {train_acc_taskA_avg: .4f}%, Average Accuracy Test (task A): {train_acc_taskB_avg: .4f}%')
            print(f'Average Accuracy Train (task B): {test_acc_taskA_avg: .4f}%, Average Accuracy Test (task B): {test_acc_taskB_avg: .4f}%')
            
        model_path = '/kaggle/working/model_' + str(epoch+1) + '.pth'
        torch.save(self.classifier.state_dict(), model_path);
            
            

    def train_epoch(self):
        train_loss_list = []
        test_loss_list = []
        train_acc_taskA_list = []
        train_acc_taskB_list = []
        test_acc_taskA_list = []
        test_acc_taskB_list = []
        
        for batch in tqdm(self.train_dataloader):
            self.optimizer.zero_grad() # ZEROING OUT THE GRADIENTS
            self.classifier.train() # TRAINING MODE
            
            # CREATING THE CLIP EMBEDDINGS
            image_list, text_list, labels_misogyny, shaming_labels, stereotype_labels, objectification_labels, violence_labels = batch
            image_list = [Image.open(f"{os.path.join(self.train_images_dir, img)}") for img in image_list] # per poterlo usare poi con CLIP

            labels_misogyny = labels_misogyny.to(self.device)
            labels_taskB = torch.stack([shaming_labels, stereotype_labels, objectification_labels, violence_labels],  dim=1).to(self.device)
    
            pred_taskA, pred_taskB = self.classifier(text_list, image_list)

            loss_A = self.loss_taskA(pred_taskA, labels_misogyny)
            loss_B = self.loss_taskB(pred_taskB, labels_taskB, reduction='mean')
            loss = (self.weight_taskA * loss_A) + (self.weight_taskB * loss_B)
            train_loss_list.append(loss)
            
            loss.backward()
            self.optimizer.step()

            accuracy_taskA = accuracy(pred_taskA, labels_misogyny, self.threshold)
            accuracy_taskB = accuracy(pred_taskB, labels_taskB, self.threshold)
            
            train_acc_taskA_list.append(accuracy_taskA)
            train_acc_taskB_list.append(accuracy_taskB)
        
        
        with torch.no_grad():
            self.classifier.eval()

            for batch in tqdm(self.test_dataloader):
                # CREATING THE CLIP EMBEDDINGS
                image_list, text_list, labels_misogyny, shaming_labels, stereotype_labels, objectification_labels, violence_labels = batch

                image_list = [Image.open(f"{os.path.join(self.test_image_dir, img)}") for img in image_list] # per poterlo usare poi con CLIP
                labels_misogyny = labels_misogyny.to(self.device)
                labels_taskB = torch.stack([shaming_labels, stereotype_labels, objectification_labels, violence_labels],  dim=1).to(self.device)
                
                pred_taskA, pred_taskB = self.classifier(text_list, image_list)

                loss_A = self.loss_taskA(pred_taskA, labels_misogyny)
                loss_B = self.loss_taskB(pred_taskB, labels_taskB, reduction='mean')
                loss = (self.weight_taskA * loss_A) + (self.weight_taskB * loss_B)
                test_loss_list.append(loss)

                accuracy_taskA = accuracy(pred_taskA, labels_misogyny, self.threshold)
                accuracy_taskB = accuracy(pred_taskB, labels_taskB, self.threshold)

                test_acc_taskA_list.append(accuracy_taskA)
                test_acc_taskB_list.append(accuracy_taskB)
            
        return train_loss_list, test_loss_list, train_acc_taskA_list, train_acc_taskB_list, test_acc_taskA_list, test_acc_taskB_list
