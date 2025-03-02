import shap
import torch
from torchvision import transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from utils import *


class OnlyTextCls(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.classifier = cls
    
    def forward(self, text_list):
        text_list = [el.item() for el in text_list]
        null_images = [Image.new('RGB', (100, 100), color=(0, 0, 0)) for _ in range(len(text_list))]
        prediction, _ = self.classifier(text_list, null_images) # for now we return only the prediction about the main task (binary one)
    
        return prediction
    
    
class OnlyImageCls(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.classifier = cls
    
    def forward(self, image_list):
        null_text = [""  for _ in range(len(image_list))]
        prediction, _ = self.classifier(null_text, image_list) # for now we return only the prediction about the main task (binary one)
        return prediction


class SingleModAnalyzer:
    def __init__(self,
                 model,
                 txt_tokenizer,
                 img_shape, # CxWxH
                 mask_token_txt="..."):
        
        self.img_shape = img_shape
        
        self.only_txt_classifier = OnlyTextCls(model)
        self.txt_masker = shap.maskers.Text(txt_tokenizer, mask_token = mask_token_txt, collapse_mask_token = False)
        self.txt_explainer = shap.Explainer(self.only_txt_classifier, masker=self.txt_masker)
        
        self.only_img_classifier = OnlyImageCls(model)
        self.img_masker = shap.maskers.Image("blur(128, 128)", (img_shape[1], img_shape[2], img_shape[0])) # image masker works with WxHxC
        self.img_explainer = shap.Explainer(self.only_img_classifier, self.img_masker)

    def txt_only_SHAP(self, txt_to_explain):
        self.only_txt_classifier.classifier.eval()
        shap_values = self.txt_explainer([txt_to_explain])
        return shap_values
    
    def img_only_SHAP(self, img_to_explain):
        resize = transforms.Resize((self.img_shape[1], self.img_shape[2]))
        img_to_explain = resize(img_to_explain).permute(1, 2, 0) # from CxWxH to WxHxC
        
        self.only_img_classifier.classifier.eval()
        
        shap_values = self.img_explainer(
            img_to_explain.unsqueeze(0), # we add a batch dimension
            max_evals=150,
            batch_size=50,
        )
        return shap_values
    
    def SHAP_single_mod(self, txt_to_explain, img_to_explain):
        print("Preparing single mode text/image explanations...")
        img_shap_val = self.img_only_SHAP(img_to_explain)
        txt_shap_val = self.txt_only_SHAP(txt_to_explain)

        print("Displaying single mode text/image explanations...")
        shap.image_plot(
            shap_values=img_shap_val.values,
            pixel_values=img_shap_val.data.numpy(),
        )

        shap_explanation = shap.Explanation(values=txt_shap_val.values.reshape(-1), feature_names=txt_shap_val.data[0])
        shap.plots.bar(shap_explanation, max_display=10)
        plt.show()

       
        
