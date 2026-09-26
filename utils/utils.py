import wandb
import torch
from torch.utils.data import Subset
from datasets import load_dataset
from pytorch_metric_learning import losses
import numpy as np
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.models import Baseline
from dataloaders.dataset import ComicsPAPDataset

CLIP_SCORES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clip_scores")


def make(config, device="cuda", use_subset=True):

    raw_dataset   = load_dataset("/home/datasets/COMIC-PAP")
    embedding_dim = getattr(config, "embedding_dim",  512)
    freeze_sbert  = getattr(config, "freeze_sbert",   True)
    freeze_resnet = getattr(config, "freeze_resnet",  True)
    clip_thr      = getattr(config, "clip_threshold", 0.0)

    NUM_WORKERS = 4

    model     = Baseline(embedding_dim=embedding_dim,
                         freeze_sbert=freeze_sbert,
                         freeze_resnet=freeze_resnet).to(device)
    tokenizer = model.sbert_tokenizer

    clip_train_path = os.path.join(CLIP_SCORES_DIR, "clip_scores_train.json")
    clip_val_path   = os.path.join(CLIP_SCORES_DIR, "clip_scores_validation.json")

    full_train_data = ComicsPAPDataset(
        raw_dataset['train'], tokenizer,
        clip_scores_path=clip_train_path if clip_thr > 0 else None,
        clip_threshold=clip_thr
    )

    val_data = ComicsPAPDataset(
        raw_dataset['validation'], tokenizer,
        clip_scores_path=None,
        clip_threshold=0.0
    )

    if use_subset:
        num_train = len(full_train_data)
        rng = np.random.default_rng(42)

        fraction   = 0.40
        n_subset   = int(num_train * fraction)
        subset_indices = rng.choice(num_train, n_subset, replace=False)
        train_data     = Subset(full_train_data, subset_indices)
        print(f"  [Subset] {int(fraction*100)}% del dataset filtrat: "
              f"{len(train_data)} mostres de {num_train} "
              f"(clip_threshold={clip_thr:.2f})")
    else:
        train_data = full_train_data
        print(f"  [Subset] Dataset complet: {len(train_data)} mostres "
              f"(clip_threshold={clip_thr:.2f})")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=config.batch_size,
        shuffle=True, pin_memory=True, num_workers=NUM_WORKERS,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=config.batch_size,
        shuffle=False, pin_memory=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.1
    )

    criterion = losses.TripletMarginLoss(
        margin=getattr(config, "margin", 0.2)
    )

    return model, train_loader, val_loader, criterion, optimizer