import shap
import torch
import numpy as np
import copy 
import math 
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from nltk.tokenize import word_tokenize


class MMSHAP:
    
    def __init__(self,
                 classifier):
        self.classifier = classifier
        self.img = None
        self.txt = None
        self.num_txt_token = None
        self.patch_size = None
        
    def display_image_text(self, shap_values):
        print(f'shap_values.shape: {shap_values.shape}')
        
        # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
        shap_values_txt = shap_values.values[0, :self.num_txt_token]
        shap_values_patches = shap_values.values[0, self.num_txt_token:]
        shap_values_img = torch.zeros(self.img.shape) # DA RIEMPIRE CON GLI SHAPLEY VALUE CALCOLATI SULLE PATCH
        data_txt = shap_values.data[0, :self.num_txt_token]
        
        print(f'shap_values_txt.shape: {shap_values_txt.shape}')
        print(f'shap_values_patches.shape: {shap_values_patches.shape}')
        print(f'shap_values_img.shape: {shap_values_img.shape}')
        print(f'self.image.shape: {self.img.shape}')
        print(f'patch_size: {self.patch_size}')
        
        row_cols = 224 // self.patch_size # 224 / 32 = 7

        # PATCHIFY THE IMAGE & SET THE PATCH TO THE RIGHT SHAPLEY VALUE (THIS MATRIX WILL THEN BE PASSED TO THE IMAGE_PLOT FUNCTION IN SHAP)
        for (i, shap_val) in enumerate(shap_values_patches):
            print(f'i: {i} - shap_val: {shap_val}')
            m = i // row_cols
            n = i % row_cols
            shap_values_img[:, m*self.patch_size:(m+1)*self.patch_size, n*self.patch_size:(n+1)*self.patch_size] = shap_val # torch.rand(3, patch_size, patch_size)  # np.random.rand()
        
        # plot shapley values for texts and images
        
        print(f"shap_values_img.unsqueeze(0): {shap_values_img.unsqueeze(0).shape}")
        print(f"pixel_values: {np.expand_dims(self.img.numpy(),0).shape}")

        shap.image_plot(
            shap_values=shap_values_img.unsqueeze(0), 
            pixel_values=np.expand_dims(self.img.numpy(),0)
        )
        
        #shap.plots.text(txt_shap_val[0], display=False)
        
        # fai qui quello del testo
        
        
    def custom_masker(self, mask, x):
        masked_X = np.copy(x).reshape(1, -1) # fai controllo per vedere se effettivamente ha una shape  e.g. (1, 15)
        mask = np.expand_dims(mask, axis=0) # same as unsqueeze(0)
        masked_X[~mask] = "UNK"
        return masked_X

    def get_model_prediction(self, x): # x must be an ndarray of strings representing the couple (perturbed_txt, perturbed_img)
        #print(x)
        self.classifier.eval()
        perturbed_imgs = []

        with torch.no_grad():
            # split up the input_ids and the image_token_ids from x (containing both appended)
            masked_txt_tokens = [input_string[:self.num_txt_token] for input_string in x]
            masked_image_tokens = [input_string[self.num_txt_token:] for input_string in x]
            perturbed_txts = [' '.join(token_list.tolist()) for token_list in masked_txt_tokens]
            
            #print(perturbed_txts)

            result = np.zeros(len(x))
            row_cols = 224 // self.patch_size # 224 / 32 = 7

            # call the model for each "new image" generated with masked features
            for i in range(len(x)):
                perturbed_img = copy.deepcopy(self.img)

                # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
                curr_masked_txt_tokens = copy.deepcopy(masked_txt_tokens[i])

                # PATCHIFY THE IMAGE
                for k in range(len(masked_image_tokens[i])):
                    if masked_image_tokens[i][k] == "UNK":  # should be the patch we want to mask
                        m = k // row_cols
                        n = k % row_cols
                        perturbed_img[:, m*self.patch_size:(m+1)*self.patch_size, n*self.patch_size:(n+1)*self.patch_size] = 0 # torch.rand(3, patch_size, patch_size)  # np.random.rand()

                perturbed_imgs.append(perturbed_img)

            outputs_taskA, _ = self.classifier(perturbed_txts, perturbed_imgs)

        return outputs_taskA

    
    def compute_mmscore(self, num_txt_tokens, shap_values):
        print(shap_values.values.shape)
        #print(shap_values.data)
        
        text_contrib = np.abs(shap_values.values[0, :num_txt_tokens]).sum()
        image_contrib = np.abs(shap_values.values[0, num_txt_tokens:]).sum()
        text_score = text_contrib / (text_contrib + image_contrib)
        image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
        return text_score, image_score
    
    def wrapper_mmscore(self, txt_to_explain, img_to_explain): #img must be a tensor of shape CxWxH
        
            
        txt_tokens = word_tokenize(txt_to_explain)
        num_txt_tokens = len(txt_tokens)
        p = int(math.ceil(np.sqrt(num_txt_tokens)))
        patch_size = 224 // p
        img_tokens = [" " for el in range(1, p**2+1)]
        txt_tokens = np.array(txt_tokens + img_tokens)

        self.img = img_to_explain
        self.txt = txt_to_explain
        self.num_txt_token = num_txt_tokens
        self.patch_size = patch_size

        explainer = shap.Explainer(self.get_model_prediction, self.custom_masker, silent=True)

        # print(txt_tokens.shape)
        # print(type(txt_tokens))
        txt_tokens = txt_tokens.reshape(1, -1)
        #print(txt_tokens)

        shap_values = explainer(txt_tokens)
        self.display_image_text(shap_values)
                    
        print(self.num_txt_token)
        print(shap_values.values.shape)
        print(shap_values.values[0, self.num_txt_token:])

        text_score, image_score = self.compute_mmscore(num_txt_tokens, shap_values)
   
        
        return text_score # image_score si ricava in automatico da text_score