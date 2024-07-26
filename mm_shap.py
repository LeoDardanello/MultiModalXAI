import shap
import torch
import numpy as np
import copy 
import math 
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from nltk.tokenize import word_tokenize


class MMSHAP:
    def __init__(self,
                 classifier,
                 max_text_features_to_visualize=15):
        self.classifier = classifier
        self.img = None
        self.txt = None
        self.num_txt_token = None
        self.patch_size = None
        self.max_text_features_to_visualize = max_text_features_to_visualize
        
    def display_image_text(self, shap_values):   
        shap_values_txt = shap_values.values[0, :self.num_txt_token]
        shap_values_patches = shap_values.values[0, self.num_txt_token:]
        shap_values_img = torch.zeros(self.img.shape)
        data_txt = shap_values.data[0, :self.num_txt_token]
        
        row_cols = self.img.shape[1] // self.patch_size # 224 / 32 = 7 (224 is the size that WE DECIDED TO USE AS EXAMPLE, you could use the shap that you like and that your images support)

        # PATCHIFY THE IMAGE & SET THE PATCH TO THE SHAP VALUE SPECIFIED (THIS MATRIX WILL THEN BE PASSED TO THE IMAGE_PLOT FUNCTION IN SHAP)
        
        for (i, shap_val) in enumerate(shap_values_patches):
            m = i // row_cols
            n = i % row_cols
            shap_values_img[:, m*self.patch_size:(m+1)*self.patch_size, n*self.patch_size:(n+1)*self.patch_size] = shap_val # torch.rand(3, patch_size, patch_size)  # np.random.rand()

        # plot shapley values for texts and images
        shap_values_img = shap_values_img.permute(1, 2, 0) # from CxWxH to WxHxC because the explainer desires this format
        self.img = self.img.permute(1, 2, 0)
        
        print("Displaying multimodal interaction for text/image explanations...")
        
        shap.image_plot(
            shap_values = [shap_values_img.unsqueeze(0).numpy()], 
            pixel_values = self.img.unsqueeze(0).numpy()
        )

        shap_explanation = shap.Explanation(values=shap_values_txt, feature_names=data_txt)
        shap.plots.bar(shap_explanation, max_display=10)
        plt.show()
        
    def custom_masker(self, mask, x):
        masked_X = np.copy(x).reshape(1, -1)
        mask = np.expand_dims(mask, axis=0)
        masked_X[~mask] = "UNK"
        return masked_X

    def get_model_prediction(self, x): # x must be an ndarray of strings representing the couple (perturbed_txt, perturbed_img)
        self.classifier.eval()
        perturbed_imgs = []

        with torch.no_grad():
            # split up the input_ids and the image_token_ids from x (containing both appended)
            masked_txt_tokens = [input_string[:self.num_txt_token] for input_string in x]
            perturbed_txts = [' '.join(token_list.tolist()) for token_list in masked_txt_tokens]
            masked_image_tokens = [input_string[self.num_txt_token:] for input_string in x]
            
            row_cols = 224 // self.patch_size # 224 / 32 = 7

            # call the model for each "new image" generated with masked features
            for i in range(len(x)):
                perturbed_img = copy.deepcopy(self.img)

                # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
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
        text_contrib = np.abs(shap_values.values[0, :num_txt_tokens]).sum()
        image_contrib = np.abs(shap_values.values[0, num_txt_tokens:]).sum()
        text_score = text_contrib / (text_contrib + image_contrib)
        image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
        return text_score, image_score
    
    def wrapper_mmscore(self, txt_to_explain, img_to_explain): # img must be a tensor of shape CxWxH
        print("Computing multimodal interaction for text/image explanations...")
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
        txt_tokens = txt_tokens.reshape(1, -1)
        shap_values = explainer(txt_tokens)
        self.display_image_text(shap_values)

        text_score, _ = self.compute_mmscore(num_txt_tokens, shap_values)
        return text_score # compute imahe_score from text_score as (1-text_Score)
