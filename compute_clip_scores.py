import io
import os
import json
import argparse
import statistics

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

DATASET_PATH = "/home/datasets/COMIC-PAP"
OUTPUT_DIR   = "clip_scores"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 32

def get_image(sample):
    sol_idx = sample["solution_index"]
    if sol_idx < 0:
        sol_idx = 0
    opt = sample["options"][sol_idx]
    if isinstance(opt, Image.Image):
        return opt.convert("RGB")
    img_bytes = opt["bytes"] if isinstance(opt, dict) else opt
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def get_first_paragraph(sample):

    caption = sample.get("previous_panel_caption", "") or ""
    paragraphs = [p.strip() for p in caption.split("\n") if p.strip()]
    return paragraphs[0] if paragraphs else ""


@torch.no_grad()
def compute_scores_batch(clip_model, clip_processor, device, images, captions):

    inputs = clip_processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    out      = clip_model(**inputs)
    img_norm = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img_norm * txt_norm).sum(dim=-1).cpu().tolist()



def process_split(split_name, clip_model, clip_processor, device, batch_size):

    print(f"\n{'='*55}")
    print(f"  Split: {split_name}")
    print(f"{'='*55}")

    split_data = load_dataset(DATASET_PATH, split=split_name)
    n_total    = len(split_data)
    print(f"  Mostres totals: {n_total}")

    scores_dict = {}   
    errors      = 0
    no_caption  = 0

    batch_images   = []
    batch_captions = []
    batch_ids      = []

    def flush_batch():
        if not batch_images:
            return
        sc = compute_scores_batch(
            clip_model, clip_processor, device,
            batch_images, batch_captions
        )
        for sid, score in zip(batch_ids, sc):
            scores_dict[sid] = round(float(score), 4)
        batch_images.clear()
        batch_captions.clear()
        batch_ids.clear()

    for sample in tqdm(split_data, desc=f"  {split_name}", total=n_total):
        sid       = sample["sample_id"]
        first_par = get_first_paragraph(sample)

        if not first_par:
            scores_dict[sid] = 0.0
            no_caption += 1
            continue

        try:
            image = get_image(sample)
        except Exception as e:
            errors += 1
            scores_dict[sid] = 0.0
            if errors <= 5:
                print(f"\n Error mostra {sid}: {e}")
            continue

        batch_images.append(image)
        batch_captions.append(first_par)
        batch_ids.append(sid)

        if len(batch_images) >= batch_size:
            flush_batch()

    flush_batch()

    all_scores     = list(scores_dict.values())
    scores_nonzero = [s for s in all_scores if s > 0.0]
    n_nonzero      = len(scores_nonzero)

    print(f"\n  Resum del split '{split_name}':")
    print(f"    Mostres processades    : {len(scores_dict)}")
    print(f"    Mostres amb caption    : {n_nonzero}")
    print(f"    Sense caption (score=0): {no_caption}")
    print(f"    Errors d'imatge        : {errors}")

    if scores_nonzero:
        print(f"\n    Estadístiques de CLIP score (primer paràgraf):")
        print(f"      Min    : {min(scores_nonzero):.4f}")
        print(f"      Max    : {max(scores_nonzero):.4f}")
        print(f"      Mitjà  : {statistics.mean(scores_nonzero):.4f}")
        print(f"      Mediana: {statistics.median(scores_nonzero):.4f}")
        print(f"      Stdev  : {statistics.stdev(scores_nonzero):.4f}")

        print(f"\n    Mostres que s'eliminarien per threshold:")
        for thr in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
            n_out = sum(1 for s in all_scores if s < thr)
            pct   = 100 * n_out / n_total
            print(f"      < {thr:.2f}: {n_out:7d} / {n_total} mostres "
                  f"({pct:.1f}%)")

    return scores_dict


def main():
    parser = argparse.ArgumentParser(
        description="Calcula CLIP score (primer paràgraf) per mostra de ComicsPAP"
    )
    parser.add_argument("--splits",     nargs="+",
                        default=["train", "validation", "test"],
                        help="Splits a processar")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Mida del batch per a CLIP")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Carpeta on guardar els JSON")
    parser.add_argument("--device",     type=str, default=None,
                        help="'cuda' o 'cpu' (per defecte: automàtic)")
    parser.add_argument("--force",      action="store_true",
                        help="Recalcula encara que el fitxer ja existeixi")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Dispositiu : {device}")
    print(f"Output dir : {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nCarregant CLIP ({CLIP_MODEL})...")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()
    print("  CLIP carregat")

    for split in args.splits:
        out_path = os.path.join(args.output_dir, f"clip_scores_{split}.json")

        if os.path.exists(out_path) and not args.force:
            print(f"\n {out_path} ja existeix → saltant "
                  f"(usa --force per recalcular)")
            continue

        scores = process_split(
            split, clip_model, clip_processor, device, args.batch_size
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"\n Guardat: {out_path}")




if __name__ == "__main__":
    main()