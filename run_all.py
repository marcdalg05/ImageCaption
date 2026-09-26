

import io
import os
import sys
import json
import time
import random
import datetime
import argparse

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

CLIP_SCORES_DIR = os.path.join(script_dir, "clip_scores")
BEST_MODEL_PATH = os.path.join(script_dir, "trained_models", "model_best.pth")

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_image(sample):
    from PIL import Image
    sol_idx = sample["solution_index"]
    if sol_idx < 0:
        sol_idx = 0
    opt = sample["options"][sol_idx]
    if isinstance(opt, Image.Image):
        return opt.convert("RGB")
    elif isinstance(opt, dict) and "bytes" in opt:
        return Image.open(io.BytesIO(opt["bytes"])).convert("RGB")
    elif isinstance(opt, bytes):
        return Image.open(io.BytesIO(opt)).convert("RGB")
    else:
        raise ValueError(f"Format desconegut: {type(opt)}")


@torch.no_grad()
def compute_scores_batch(clip_model, clip_processor, device, images, captions):
    inputs = clip_processor(
        text=captions, images=images,
        return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    out      = clip_model(**inputs)
    img_norm = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img_norm * txt_norm).sum(dim=-1).cpu().tolist()


def compute_clip_scores_for_split(split_name, clip_model, clip_processor,
                                   device, batch_size=32):
    from datasets import load_dataset

    out_path = os.path.join(CLIP_SCORES_DIR, f"clip_scores_{split_name}.json")

    log(f"  Carregant split '{split_name}'...")
    data   = load_dataset("/home/datasets/COMIC-PAP", split=split_name)
    total  = len(data)
    log(f"  {total} mostres")

    img0, cap0 = get_image(data[0]), str(data[0].get("previous_panel_caption","") or "")
    s0 = compute_scores_batch(clip_model, clip_processor, device, [img0], [cap0[:200]])[0]
    log(f" Score de prova (mostra 0): {s0:.4f} {'MES BAIX DE L ESPERAT' if s0 < 0.05 else '(OK)'}")

    scores_dict    = {}
    images_batch   = []
    captions_batch = []
    indices_batch  = []
    errors         = 0
    t_start        = time.time()

    for i, sample in enumerate(data):
        try:
            img = get_image(sample)
            cap = str(sample.get("previous_panel_caption", "") or "")[:500]
            images_batch.append(img)
            captions_batch.append(cap)
            indices_batch.append(i)
        except Exception as e:
            errors += 1
            scores_dict[str(i)] = 0.0
            if errors <= 3:
                log(f"  Error mostra {i}: {e}")
            continue

        if len(images_batch) >= batch_size:
            scores = compute_scores_batch(clip_model, clip_processor, device,
                                          images_batch, captions_batch)
            for idx, sc in zip(indices_batch, scores):
                scores_dict[str(idx)] = round(float(sc), 4)
            images_batch = []; captions_batch = []; indices_batch = []

        if (i + 1) % 500 == 0:
            elapsed  = time.time() - t_start
            pct      = 100 * (i + 1) / total
            eta_s    = elapsed / (i + 1) * (total - i - 1)
            eta_min  = eta_s / 60
            vals_so_far = [v for v in scores_dict.values() if v > 0]
            mean_sc  = sum(vals_so_far) / len(vals_so_far) if vals_so_far else 0
            log(f"  [{pct:5.1f}%] {i+1}/{total} — score mitjà fins ara: {mean_sc:.4f} — ETA: {eta_min:.0f} min")

    if images_batch:
        scores = compute_scores_batch(clip_model, clip_processor, device,
                                      images_batch, captions_batch)
        for idx, sc in zip(indices_batch, scores):
            scores_dict[str(idx)] = round(float(sc), 4)

    vals = [v for v in scores_dict.values() if v > 0]
    if vals:
        log(f"  Estadístiques '{split_name}':")
        log(f"    Min={min(vals):.4f}  Max={max(vals):.4f}  "
            f"Mitjà={sum(vals)/len(vals):.4f}  Errors={errors}")
        for thr in [0.10, 0.15, 0.20, 0.25]:
            count = sum(1 for v in vals if v < thr)
            log(f"    < {thr:.2f}: {count} mostres ({100*count/len(vals):.1f}%)")

    os.makedirs(CLIP_SCORES_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scores_dict, f)
    log(f"  Scores guardats a: {out_path}")
    return out_path


def step1_clip_scores(device, batch_size=32, force=False):
    log("=" * 60)
    log("PAS 1: CÀLCUL DE CLIP SCORES")
    log("=" * 60)

    splits_needed = []
    for split in ["train", "validation", "test"]:
        path = os.path.join(CLIP_SCORES_DIR, f"clip_scores_{split}.json")
        if force or not os.path.exists(path):
            splits_needed.append(split)
        else:
            with open(path) as f:
                data = json.load(f)
            nonzero = sum(1 for v in data.values() if float(v) > 0)
            if nonzero == 0:
                log(f"   {path} té tots els scores a 0 → recalculant")
                splits_needed.append(split)
            else:
                log(f"  {path} ja existeix ({nonzero} scores no-zero) → saltant")

    if not splits_needed:
        log("  Tots els CLIP scores ja calculats correctament.")
        return

    log(f"  Carregant CLIP (openai/clip-vit-base-patch32)...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    log("  CLIP carregat")

    for split in splits_needed:
        log(f"\n  — Processant split: {split} —")
        compute_clip_scores_for_split(split, clip_model, clip_processor,
                                       device, batch_size)

    del clip_model
    torch.cuda.empty_cache()
    log("\n  PAS 1 COMPLETAT. Memòria GPU alliberada.")



def step2_sweep(device):
    import wandb
    from utils.utils import make
    from train import train

    log("=" * 60)
    log("PAS 2: SWEEP WANDB (SBERT + ResNet50 + TripletLoss)")
    log("=" * 60)

    torch.backends.cudnn.deterministic = True
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    sweep_config = {
        "name":   "comics-sbert-triplet-clip-filtered",
        "method": "bayes",
        "metric": {"name": "best_R@1", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 2, "eta": 2},
        "parameters": {
            "learning_rate":   {"values": [1e-4, 2e-5, 5e-5]},
            "batch_size":      {"values": [64, 128, 256]},
            "embedding_dim":   {"values": [512]},
            "margin":          {"values": [0.1, 0.2, 0.3]},
            "clip_threshold":  {"values": [0.0, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25]},
            "freeze_sbert":    {"value": True},
            "freeze_resnet":   {"value": True},
            "epochs":          {"values": [5, 10]},
            "dataset":         {"value": "ComicsPAP"},
            "architecture":    {"value": "SentenceBERT-ResNet50-Triplet-CLIPfilter"},
        }
    }

    best_r1_global = [0.0] 

    def model_pipeline():
        with wandb.init() as run:
            config = wandb.config
            log(f"\n{'='*60}")
            log(f"Run: {run.name}")
            log(f"  lr={config.learning_rate}  bs={config.batch_size}  "
                f"margin={config.margin}  clip_thr={config.clip_threshold}  "
                f"epochs={config.epochs}")
            log(f"{'='*60}")

            model, train_loader, val_loader, criterion, optimizer = make(
                config, device=device
            )

            best_r1_run = train(
                model=model, loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer,
                config=config, device=device,
            )

            if best_r1_run > best_r1_global[0]:
                best_r1_global[0] = best_r1_run
                os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                log(f"*** NOU MILLOR MODEL! R@1={best_r1_global[0]:.4f} "
                    f"guardat a {BEST_MODEL_PATH} ***")

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Comics-Retrieval-millores")
    log(f"  Sweep creat: {sweep_id}")
    log(f"  Veieu el progrés a: https://wandb.ai/javiermartinez18-universitat-aut-noma-de-barcelona/Comics-Retrieval-millores")
    wandb.agent(sweep_id, function=model_pipeline, count=30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_clip",   action="store_true",
                        help="Salta el càlcul de CLIP scores (usa els existents)")
    parser.add_argument("--force_clip",  action="store_true",
                        help="Força recàlcul de CLIP scores encara que existeixin")
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Batch size per al càlcul de CLIP scores")
    parser.add_argument("--device",      type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    log(f"Dispositiu: {device}")
    log(f"Inici: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Script dir: {script_dir}")

    if not args.skip_clip:
        step1_clip_scores(device, batch_size=args.batch_size, force=args.force_clip)
    else:
        log("CLIP scores: saltant (--skip_clip actiu)")

    step2_sweep(device)

    log(f"\nFet! {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()