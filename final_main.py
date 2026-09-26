import os
import sys
import random
import numpy as np
import torch
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils.utils import make
from train import train

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_dict = {
    "learning_rate":  1e-4,
    "batch_size":     64,
    "embedding_dim":  512,
    "margin":         0.4,
    "clip_threshold": 0.29,
    "freeze_sbert":   True,   
    "freeze_resnet":  True,  
    "epochs":         150,
    "dataset":        "ComicsPAP",
    "architecture":   "ResNet50-SBERTcongelat-TripletLoss",
    "weight_decay":   0.1, 
}

SAVE_PATH = os.path.join(script_dir, "trained_models", "model_final_150ep.pth")


if __name__ == "__main__":
    wandb.login()

    with wandb.init(
        project="Comics-Retrieval-Final",
        name="firm-sweep-24_150epochs",
        config=config_dict,
        tags=["final", "best-hparams", "150ep"],
    ) as run:
        config = wandb.config

        print(f"\n{'='*60}")
        print(f"  RUN FINAL — millors hiperparàmetres (firm-sweep-24)")
        print(f"{'='*60}")
        print(f"  learning_rate  : {config.learning_rate}")
        print(f"  batch_size     : {config.batch_size}")
        print(f"  embedding_dim  : {config.embedding_dim}")
        print(f"  margin         : {config.margin}")
        print(f"  clip_threshold : {config.clip_threshold}")
        print(f"  epochs         : {config.epochs}")
        print(f"  freeze_sbert   : {config.freeze_sbert}")
        print(f"  weight_decay   : {config.weight_decay}")
        print(f"{'='*60}\n")

        model, train_loader, val_loader, criterion, optimizer = make(
            config, device=device
        )

        best_r1 = train(
            model=model,
            loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save({
            "config":   dict(config),
            "model":    model.state_dict(),
            "best_r1":  best_r1,
        }, SAVE_PATH)
        print(f"\n Model final guardat a: {SAVE_PATH}")
        print(f"   Millor R@1 assolit: {best_r1:.4f}")
