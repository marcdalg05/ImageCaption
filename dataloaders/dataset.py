import json
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class ComicsPAPDataset(Dataset):

    def __init__(self, hf_dataset, tokenizer, max_length=128,
                 clip_scores_path=None, clip_threshold=0.0):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        if clip_scores_path and os.path.exists(clip_scores_path) and clip_threshold > 0.0:
            with open(clip_scores_path, encoding="utf-8") as f:
                clip_scores = json.load(f)
            self.valid_indices = [
                i for i in range(len(hf_dataset))
                if float(clip_scores.get(hf_dataset[i]["sample_id"], 1.0)) >= clip_threshold
            ]
            n_total   = len(hf_dataset)
            n_valid   = len(self.valid_indices)
            n_removed = n_total - n_valid
            print(f"  [CLIP filter] threshold={clip_threshold:.2f} → "
                  f"conservem {n_valid}/{n_total} mostres "
                  f"(eliminades {n_removed}, {100*n_removed/n_total:.1f}%)")
        else:
            self.valid_indices = list(range(len(hf_dataset)))
            if clip_threshold > 0.0:
                print(f"  [CLIP filter] AVÍS: no s'ha trobat {clip_scores_path}, "
                      f"s'usa el dataset complet sense filtrar")

        self.ds = hf_dataset

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item    = self.ds[self.valid_indices[idx]]
        sol_idx = item['solution_index']
        if sol_idx < 0:
            sol_idx = 0

        image   = self.transform(item['options'][sol_idx].convert('RGB'))
        caption = item.get('previous_panel_caption', '') or ''

        encoding = self.tokenizer(
            str(caption),
            padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        return {
            'image':          image,
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }