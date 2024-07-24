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



if __name__ == "__main__":
    ###########################
    ### JSON FILE GENERATION ###
    ###########################

    #images_path = "/kaggle/input/dataset-wow/MAMI DATASET/MAMI DATASET/training/TRAINING"
    """
    file_and_dest = [('train_image_text.tsv','train_image_text.json'),
                    ('test_image_text.tsv','test_image_text.json')]

    generate_json_file(file_and_dest)
    """

    """>
    ######################
    ### MODEL TRAINING ###
    ######################
    model_trainer = Trainer("/kaggle/input/dataset-wow/MAMI DATASET/MAMI DATASET/training/TRAINING",# train_images_dir
                            "/kaggle/input/dataset-wow/MAMI DATASET/MAMI DATASET/test", #test_images_dir
                            "/kaggle/working/train_image_text.json",
                            "/kaggle/working/test_image_text.json",
                            batch_size=64,
                            num_epochs=3) # json_file as data source


    model_trainer.train_model()
    
    """
    
    #############################
    ### EXPLANATION GENERATION ###
    ##############################


    # loading the model to explain
    checkpoint = torch.load('C:\\Users\\chito\\Desktop\\model_3.pth', map_location=torch.device('cpu'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = MisogynyCls(5).to(device)
    classifier.load_state_dict(checkpoint)
    
    txt_tokenizer = custom_word_tokenizer # taken from utils.py # text masker definition
    txt_to_explain = "HELLO, PRINCE! WILL YOU RAPE ME AND LIVE HAPPILY EVER AFTER WITH ME?"

    image = ToTensor()(Image.open("./img_demo.jpg"))
    img_to_explain = torch.clamp(image, min=0.0, max=np.float64(1))
    img_to_explain = (img_to_explain * 255).byte() # returns a tensor to avoid clamp errors

    """
    data = MultimodalDataset("C:\\Users\\chito\\Desktop\\dataset_xai\\MAMI DATASET\\training\\TRAINING", "train_image_text.json")
    data = DataLoader(data, 100, shuffle=True, pin_memory=True)
    images, texts, _, _, _, _, _ = next(iter(data))
    #images = [ToTensor()(Image.open(f"{os.path.join('C:\\Users\\chito\\Desktop\\dataset_xai\\MAMI DATASET\\training\\TRAINING', img)}")) for img in images if exists((f"{os.path.join('C:\\Users\\chito\\Desktop\\dataset_xai\\MAMI DATASET\\training\\TRAINING', img)}"))]
    images = [print(f"{os.path.join('C:\\Users\\chito\\Desktop\\dataset_xai\\MAMI DATASET\\training\\TRAINING', img)}") for img in images]
    img_to_explain = torch.clamp(images[27], min=0.0, max=np.float64(1)) # 27 is a random index
    img_to_explain = (img_to_explain * 255).byte() # returns a tensor to avoid clamp errors

    txt_to_explain= texts[27]

    print(f'initial shape: {len(img_to_explain)}')9
    """

    explainer=DMSBE(classifier, txt_tokenizer, (3, 224, 224))
    # .explain() takes a list of texts and a list of images,if explaining a single sample pass a list with one element
    explainer.explain([txt_to_explain], [img_to_explain])
