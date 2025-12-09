import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from t5_generator import MedBLIPReportGeneratorT5
from dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_dataloaders(train_config):
    traindata = ImageTextContrastiveDataset(csv_path=r"E:\data_nour\train_all.csv")
    trainloader = DataLoader(
        traindata,
        batch_size=train_config['train_batch_size'],
        collate_fn=ImageTextContrastiveCollator(),
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )

    valdata = ZeroShotImageDataset(csv_path=r"E:\data_nour\val_all.csv")
    valloader = DataLoader(
        valdata,
        batch_size=train_config['eval_batch_size'],
        collate_fn=ZeroShotImageCollator(),
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    return trainloader, valloader


    
def train_model(model, trainloader, valloader, config, output_path):
    model.cuda()
    trainer = Trainer()
    trainer.train(
        model=model,
        dataloader=trainloader,
        eval_dataloader=valloader,
        warmup_ratio=config['warmup'],
        epochs=config['num_epochs'],
        optimizer_params={'lr': config['lr']},
        output_path=output_path,
        weight_decay=config['weight_decay'],
        accumulation_steps=config['accumulation_steps'],  
        use_amp=False,
        eval_steps=config['eval_steps'],  # Ajout de cette ligne
    )

def main():
    set_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_config = {
        'num_epochs': 50,
        'use_amp':False,
        'warmup': 0.1,
        'lr': 5e-5 ,
        'lr_scheduler': 'linear',  
        'warmup_steps': 1000,                   
        'weight_decay': 0.01,          
        'train_batch_size': 5,        
        'temperature': 0.7,            
        'repetition_penalty': 1.5,     
        'max_new_tokens': 300,
        'gradient_clip': 1.0,
        'eval_batch_size': 6,
        'eval_steps': 100,
        'gradient_accumulation_steps': 2,
        'accumulation_steps': 4,
        'save_steps': 1000,
    }

    trainloader, valloader = get_dataloaders(train_config)
    t5 = True
    biomedlm = False

    if t5:
        print("Training with T5...")
        model = MedBLIPReportGeneratorT5(t5_model="google/flan-t5-base")
        train_model(model, trainloader, valloader, train_config, "E:/data_nour/checkpoints/vision_text_pretrain/t5")

    if biomedlm:
        print("Training with BioMedLM...")
        model = MedBLIPReportGenerator(lm_model="stanford-crfm/BioMedLM")
        train_model(model, trainloader, valloader, train_config, "E:/data_nour/checkpoints/vision_text_pretrain/biomedlm")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()