import os
import torch
from dataset import ZeroShotImageDataset, ZeroShotImageCollator
from torch.utils.data import DataLoader
from t5_new_baseline import MedBLIPReportGeneratorT5
from trainer_new_baseline import Trainer

def load_test_set(config):
    test_data = ZeroShotImageDataset(csv_path=r"E:\data_nour\test_all.csv")
    return DataLoader(
        test_data,
        batch_size=config['eval_batch_size'],
        collate_fn=ZeroShotImageCollator(),
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

def evaluate_final_model(model_checkpoint, output_dir):
    eval_config = {
        'eval_batch_size': 6,
        'max_new_tokens': 500,
        'temperature': 0.7,
        'repetition_penalty': 1.5
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Chargement du modèle depuis {model_checkpoint} ===")
    
    model = MedBLIPReportGeneratorT5(t5_model="google/flan-t5-base")
    try:
        model.load_state_dict(torch.load(model_checkpoint))
    except:
        model = torch.load(model_checkpoint)
    model.to(device)
    model.eval()
    
    test_loader = load_test_set(eval_config)
    
    print("=== Début de l'évaluation sur le test set ===")
    trainer = Trainer()
    results = trainer.evaluate(model, test_loader)
    
    # 5. Sauvegarde des résultats
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "final_test_metrics_baseline_newmodif12.txt")
    
    with open(result_file, 'w') as f:
        f.write("=== Résultats finaux ===\n")
        f.write(f"Modèle évalué: {model_checkpoint}\n\n")
        f.write("Métriques:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\n=== Résultats sauvegardés dans {result_file} ===")
    print("Détail des métriques:")
    for metric, value in results.items():
        print(f"- {metric}: {value:.4f}")

if __name__ == "__main__":
    CHECKPOINT_PATH = "E:/data_nour/checkpoints/vision_text_pretrain/t5/epoch_49.pth"  
    OUTPUT_DIR = "E:/data_nour/final_evaluation_results" 
    evaluate_final_model(CHECKPOINT_PATH, OUTPUT_DIR)