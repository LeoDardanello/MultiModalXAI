import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from misogyny_trainer import *
from multimodal_explainer import *
from utils import *

if __name__ == "__main__":
    ###########################
    ### JSON FILE GENERATION ###
    ###########################

    #images_path = "/kaggle/input/dataset-wow/MAMI DATASET/MAMI DATASET/training/TRAINING"

    file_and_dest = [('/kaggle/input/dataset-wow/train_image_text.tsv','/kaggle/working/train_image_text.json'),
                    ('/kaggle/input/dataset-wow/test_image_text.tsv','/kaggle/working/test_image_text.json')]

    generate_json_file(file_and_dest)
    

    """
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
    checkpoint = torch.load('/kaggle/input/model-params/model_3.pth', map_location=torch.device('cpu'))
    classifier = MisogynyCls(5)
    classifier.load_state_dict(checkpoint)

    # text masker definition
    txt_tokenizer = custom_word_tokenizer # taken from utils.py

    data = MultimodalDataset("/kaggle/input/dataset-wow/MAMI DATASET/MAMI DATASET/training/TRAINING", "/kaggle/working/train_image_text.json")
    data = DataLoader(data, 100, shuffle=True, pin_memory=True)
    images, texts, _, _, _, _, _ = next(iter(data))
    images = [ToTensor()(Image.open(f"{os.path.join('/kaggle/input/dataset-wow/MAMI DATASET/MAMI DATASET/training/TRAINING', img)}")) for img in images]
    img_to_explain = images[27]
    txt_to_explain = texts[27]

    analyzer = SingleModAnalyzer(classifier,
                                txt_tokenizer,
                                (3, 440, 440), # CxWxH
                                mask_token_txt="...") # da sviluppare l'img_token customizzabile

    analyzer.SHAP_single_mod(txt_to_explain, img_to_explain, "/kaggle/working/results.html")