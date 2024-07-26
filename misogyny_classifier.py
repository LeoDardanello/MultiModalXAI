import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class MisogynyCls(nn.Module):
    def __init__(self, num_linear_layers, task_a_out=1, task_b_out=4, input_dim=1024, hidden_dim=512, drop_value=0.2):
        super().__init__()
        self.head_task_a = nn.Linear(hidden_dim, task_a_out)
        self.head_task_b = nn.Linear(hidden_dim, task_b_out)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Check if CUDA is available
        
        # Pretrained CLIP loading...
        if torch.cuda.is_available():
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map='cuda')
        else:   
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


        self.layers = nn.ModuleList()

        for i in range(num_linear_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(drop_value))
            self.layers.append(nn.ReLU())

    def forward(self, text_list, image_list):
        clip_inputs = self.clip_processor(text=text_list, images=image_list, return_tensors="pt", padding=True, truncation=True)
        clip_inputs['input_ids'] = clip_inputs['input_ids'].to(self.device)
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].to(self.device)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].to(self.device)
        clip_outputs = self.clip_model(**clip_inputs)
        
        x = torch.cat([clip_outputs['text_embeds'], clip_outputs['image_embeds']], dim=1).to(self.device) # model input is the concatenation of the two modalities !
            
        for layer in self.layers:
            x = layer(x)
            
        pred_taskA = self.sigmoid(self.head_task_a(x))
        pred_taskB = self.sigmoid(self.head_task_b(x))
        
        return pred_taskA.squeeze(1), pred_taskB