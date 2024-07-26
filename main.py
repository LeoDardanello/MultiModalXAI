import torch
from DMSBE import *
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from misogyny_trainer import *
from multimodal_explainer import *
from utils import *
from mm_shap import *
import sys


if __name__ == "__main__":
    checkpoint = torch.load('C:\\Users\\chito\\Desktop\\model_3.pth', map_location=torch.device('cpu'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = MisogynyCls(5).to(device)
    classifier.load_state_dict(checkpoint)

    txt_tokenizer = custom_word_tokenizer 

    if len(sys.argv) == 1: # only the program name, then take the image from the github folder for the demo
        txt_to_explain = "HELLO, PRINCE! WILL YOU RAPE ME AND LIVE HAPPILY EVER AFTER WITH ME?"
        image = ToTensor()(Image.open("./images/img_demo.jpg"))
        
    else:
        image = ToTensor()(Image.open(sys.argv[1])) 
        txt_to_explain = sys.argv[2] 

    img_to_explain = torch.clamp(image, min=0.0, max=np.float64(1))
    img_to_explain = (img_to_explain * 255).byte() # returns a tensor to avoid clamp errors

    explainer=DMSBE(classifier, txt_tokenizer, (3, 224, 224)) # (3, 224, 224) is the shap that we want our explainer to work on... 
                                                              # this will be the size to which the input image will be resized
    
    # .explain() takes a list of texts and a list of images, if explaining a single sample pass a list with one element
    explainer.explain([txt_to_explain], [img_to_explain])

