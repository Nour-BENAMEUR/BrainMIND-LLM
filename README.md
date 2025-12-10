# BrainMIND-LLM: An AI-Powered Multimodal Large Language Model for Neurodegenerative Disease Diagnosis and Clinical Report Generation
## Dataset
We provide results Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset, The data is available on their website via this link [ (http://adni.loni.usc.edu )].
## Description
In this work we propose a novel BrainMIND-LLM framework, a multimodal LLM that integrates neuroimaging and clinical data. The proposed methodology involves data preprocessing to standardize MRI and PET scans, clinical feature embedding, and the design of a robust fusion architecture combining a 3D Vision Transformer (ViT) encoder with a T5-based language model. This design enables the system to capture both structural brain alterations and clinical context allowing the model to generate coherent and clinically informed diagnostic narratives. Experimental evaluation shows that BrainMIND-LLM generates reports well aligned with expert references, achieving a BERTScore of 0.9133, ROUGE-L F1 score of 0.5068, and strong n-gram overlap with BLEU-1 to BLEU-4 scores ranging from 0.5917 to 0.1770. It also obtains a METEOR score of 0.4708, confirming both semantic fidelity and fluency.
<img width="1022" height="541" alt="Image" src="https://github.com/user-attachments/assets/dc8f90b5-e501-4b9d-91f4-578dde2f3690" />
## Data preparation
Run the dataset.py notebook to load, clean, and organize the dataset for training.
## 3D vision transformer
Execute the eva_vit_3d.py notebook to initialize and configure the 3D Vision Transformer architecture.
## T5_generator model
Run the t5_generator.py notebook to load the T5 model and prepare it for report generation.
## Training of the model 
Launch the run_medblip_.py notebook to train the multimodal framework using the prepared data.
## Evaluation of the model
Use the test_eval.py notebook to compute performance metrics and analyze model outputs.
