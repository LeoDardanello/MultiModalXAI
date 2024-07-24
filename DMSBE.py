from multimodal_explainer import *
from mm_shap import *

class DMSBE():
    def __init__(self,
                 model, # TRAINED model
                 txt_tokenizer, # Tokenizer for text
                 img_shape, # CxWxH
                 token_for_text_masking="..."):
        
        self.model = model
        self.txt_tokenizer = txt_tokenizer
        self.img_shape = img_shape
        self.token_for_text_masking = token_for_text_masking
        self.disentagled_modalities_explainer = SingleModAnalyzer(self.model,
                                                        self.txt_tokenizer ,
                                                        self.img_shape, # CxWxH
                                                        self.token_for_text_masking)
        self.multimodal_interaction_explainer = MMSHAP(self.model)

    def explain(self, txt_to_explain, img_to_explain):
        resize = transforms.Resize((self.img_shape[1], self.img_shape[2]))

        mmscore_list= []
        
        for i in range(len(txt_to_explain)):
            byte_image = torch.clamp(img_to_explain[i], min=0.0, max=np.float64(1))
            byte_image = (byte_image * 255).byte()
            resized_img_to_explain = resize(byte_image)
            self.disentagled_modalities_explainer.SHAP_single_mod(txt_to_explain[i], resized_img_to_explain, "single_mode")
            t_shap=self.multimodal_interaction_explainer.wrapper_mmscore(txt_to_explain[i], resized_img_to_explain)
            mmscore_list.append(t_shap)
        print(f"Mean Text Modality Contribution (MM-SHAP) among {i+1} sample: {np.mean(np.array(mmscore_list))} +/- {np.std(np.array(mmscore_list))}")
        print(f"Mean Image Modality Contribution (MM-SHAP) among {i+1} sample: {1-np.mean(np.array(mmscore_list))} +/- {np.std(np.array(mmscore_list))}")

    def only_disentagled_modalities_explain(self, txt_to_explain, img_to_explain):
        resize = transforms.Resize((self.img_shape[1], self.img_shape[2]))

        for i in range(len(txt_to_explain)):
            byte_image = torch.clamp(img_to_explain[i], min=0.0, max=np.float64(1))
            byte_image = (byte_image * 255).byte()
            resized_img_to_explain = resize(byte_image)
            self.disentagled_modalities_explainer.SHAP_single_mod(txt_to_explain[i], resized_img_to_explain, "single_mod")

    def only_multimodal_interaction_explain(self, txt_to_explain, img_to_explain):
        resize = transforms.Resize((self.img_shape[1], self.img_shape[2]))
        mmscore_list= []
        for i in range(len(txt_to_explain)):
            byte_image = torch.clamp(img_to_explain[i], min=0.0, max=np.float64(1))
            byte_image = (byte_image * 255).byte()
            resized_img_to_explain = resize(byte_image)
            t_shap=self.multimodal_interaction_explainer.wrapper_mmscore(txt_to_explain[i], resized_img_to_explain)
            mmscore_list.append(t_shap)
        print(f"Mean Text Modality Contribution (MM-SHAP) among {i+1} sample: {np.mean(np.array(mmscore_list))} +/- {np.std(np.array(mmscore_list))}")
        print(f"Mean Image Modality Contribution (MM-SHAP) among {i+1} sample: {1-np.mean(np.array(mmscore_list))} +/- {np.std(np.array(mmscore_list))}")