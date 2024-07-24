# MultiModalXAI
Project for the Explanaible and Trustworthy AI course @ Polito 2023/2024:

# Disentangled Multi modal Shap Based Explanations for Vision Language Classifiers


| ![Alt text](./images/single_mode_img.png) <br> <h3 style="text-align:center;">Single modality interaction image</h3> | ![Alt text](./images/multi_mode_img.png) <br> <h3 style="text-align:center;">Multi modality interaction image</h3> |
|-------------------------------------------|-------------------------------------------|
| ![Alt text](./images/single_mode_txt.png) <br> <h3 style="text-align:center;">Single modality interaction text</h3> | ![Alt text](./images/multi_mode_txt.png) <br> <h3 style="text-align:center;">Multi modality interaction text</h3> |

## Explanation method

Our explanation method aims to explain the contribution of features from different modalities regarding vision-language classification models.<br>

We exploit a perturbation-based feature attribution method like SHAP for computing the attributes for the single modalities and the interaction of the two. This mechanism is inspired to DIME, with the difference that our approach is **performance-agnostic** and takes into account the3 problem of the **unimodal collapse**.<br>

Other than the misogyny classifier, It can be adapted to any vision-languange classifier by modifying the function that generates the prediction and its inputs (it is supposed a list of images and texts as input to the model...) for both the single and the interactive modality. In case of single modality feature attribution, we create a sub-model which is a wrapper for the actual classifier and masks with default value the current masked modality.
```
def get_model_prediction(self, x) # in mm_shap.py
class OnlyTextCls # from multimodal_explainer.py
class OnlyImageCls # in multimodal_explainer.py

```

## Setup Instructions
To set up the the environment for this project, follow these steps:

```
pip install -r requirements.txt
```

## Demo 
In order to reproduce the results showed at the beginning of the README file, produced from the trained Misogyny classifier, type the following code with no arguments:

```
python ./main.py
```

If instead you'd like to test your own meme (always with our Misogyny classifier), type the following code:

```
python ./main.py "path_to_image_meme" "text_meme"
```

The result will show the various plots for the single modalities explanation and for the multi modal one.

## Contributions/Reference

@incollection{NIPS2017_7062,<br>
title = {***A Unified Approach to Interpreting Model Predictions***},<br>
author = {Lundberg, Scott M and Lee, Su-In},<br>
booktitle = {Advances in Neural Information Processing Systems 30},<br>
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},<br>
pages = {4765--4774},<br>
year = {2017},<br>
publisher = {Curran Associates, Inc.},<br>
url = {http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}<br>
}<br><br><br>


@inproceedings{parcalabescu-frank-2023-mm,<br>
    title = "**{MM}-{SHAP}: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models {\&} Tasks**",<br>
    author = "Parcalabescu, Letitia  and
      Frank, Anette",<br>
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",<br>
    month = jul,<br>
    year = "2023",<br>
    address = "Toronto, Canada",<br>
    publisher = "Association for Computational Linguistics",<br>
    url = "https://aclanthology.org/2023.acl-long.223",<br>
    doi = "10.18653/v1/2023.acl-long.223",<br>
    pages = "4032--4059",<br>
    abstract = "Vision and language models (VL) are known to exploit unrobust indicators in individual modalities (e.g., introduced by distributional biases) instead of focusing on relevant information in each modality. That a unimodal model achieves similar accuracy on a VL task to a multimodal one, indicates that so-called unimodal collapse occurred. However, accuracy-based tests fail to detect e.g., when the model prediction is wrong, while the model used relevant information from a modality.Instead, we propose MM-SHAP, a performance-agnostic multimodality score based on Shapley values that reliably quantifies in which proportions a multimodal model uses individual modalities. We apply MM-SHAP in two ways: (1) to compare models for their average degree of multimodality, and (2) to measure for individual models the contribution of individual modalities for different tasks and datasets.Experiments with six VL models {--} LXMERT, CLIP and four ALBEF variants {--} on four VL tasks highlight that unimodal collapse can occur to different degrees and in different directions, contradicting the wide-spread assumption that unimodal collapse is one-sided. Based on our results, we recommend MM-SHAP for analysing multimodal tasks, to diagnose and guide progress towards multimodal integration. Code available at https://github.com/Heidelberg-NLP/MM-SHAP.",<br>
}<br><br><br>


**DIME: Fine-grained Interpretations of Multimodal Models via Disentangled Local Explanations**
Yiwei Lyu, Paul Pu Liang, Zihao Deng, Ruslan Salakhutdinov, Louis-Philippe Morency.






