
import io
import os
import json
import argparse
import statistics
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

DATASET_PATH = "/home/datasets/COMIC-PAP"
OUTPUT_DIR   = "alignment_analysis"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 64

def load_image(opt):
    if isinstance(opt, Image.Image):
        return opt.convert("RGB")
    img_bytes = opt["bytes"] if isinstance(opt, dict) else opt
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def get_paragraphs(sample):
    caption = sample.get("previous_panel_caption", "") or ""
    return [p.strip() for p in caption.split("\n") if p.strip()]


@torch.no_grad()
def compute_scores_batch(clip_model, clip_processor, device, images, captions):
    inputs = clip_processor(
        text=captions, images=images,
        return_tensors="pt", padding=True,
        truncation=True, max_length=77
    ).to(device)
    out      = clip_model(**inputs)
    img_norm = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img_norm * txt_norm).sum(dim=-1).cpu().tolist()


def compute_paragraph_image_matrix(clip_model, clip_processor, device,
                                   images, paragraphs, batch_size):

    n_pars  = len(paragraphs)
    n_imgs  = len(images)

    all_images   = []
    all_captions = []
    pair_index   = []

    for par_idx, par in enumerate(paragraphs):
        for img_idx, img in enumerate(images):
            all_images.append(img)
            all_captions.append(par)
            pair_index.append((par_idx, img_idx))

    matrix = [[0.0] * n_imgs for _ in range(n_pars)]

    for start in range(0, len(all_images), batch_size):
        end     = min(start + batch_size, len(all_images))
        scores  = compute_scores_batch(
            clip_model, clip_processor, device,
            all_images[start:end], all_captions[start:end]
        )
        for (par_idx, img_idx), score in zip(pair_index[start:end], scores):
            matrix[par_idx][img_idx] = round(float(score), 4)

    return matrix

def process_split(split_name, clip_model, clip_processor, device,
                  batch_size, save_all=False):
    print(f"\n{'='*60}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")

    split_data = load_dataset(DATASET_PATH, split=split_name)
    n_total    = len(split_data)
    print(f"  Mostres: {n_total}\n")

    mismatches  = []
    all_results = []
    errors      = 0

    n_no_caption      = 0
    n_correct         = 0
    n_mismatch        = 0
    clip_scores_sol   = []
    clip_scores_pred  = []

    for sample in tqdm(split_data, desc=f"  {split_name}", total=n_total):
        sid        = sample["sample_id"]
        sol_idx    = sample["solution_index"]
        if sol_idx < 0:
            sol_idx = 0
        options    = sample["options"]
        paragraphs = get_paragraphs(sample)

        if not paragraphs:
            n_no_caption += 1
            continue

        # Carregar totes les imatges de les opcions
        try:
            images = [load_image(opt) for opt in options]
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError imatge mostra {sid}: {e}")
            continue

        try:
            matrix = compute_paragraph_image_matrix(
                clip_model, clip_processor, device,
                images, paragraphs, batch_size
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError CLIP mostra {sid}: {e}")
            continue


        n_imgs       = len(images)
        mean_scores  = [
            round(statistics.mean(matrix[par_idx][img_idx]
                                  for par_idx in range(len(paragraphs))), 4)
            for img_idx in range(n_imgs)
        ]

        clip_pred_idx   = mean_scores.index(max(mean_scores))
        score_solution  = mean_scores[sol_idx]
        score_predicted = mean_scores[clip_pred_idx]
        is_correct      = (clip_pred_idx == sol_idx)

        if is_correct:
            n_correct += 1
        else:
            n_mismatch += 1

        clip_scores_sol.append(score_solution)
        clip_scores_pred.append(score_predicted)

        record = {
            "sample_id":        sid,
            "split":            split_name,
            "solution_index":   sol_idx,
            "clip_pred_index":  clip_pred_idx,
            "is_correct":       is_correct,
            "n_options":        n_imgs,
            "n_paragraphs":     len(paragraphs),
            "mean_scores":      mean_scores,
            "score_solution":   score_solution,
            "score_predicted":  score_predicted,
            "score_gap":        round(score_predicted - score_solution, 4),
            "paragraph_scores_matrix": matrix,
            "first_paragraph":  paragraphs[0][:300],
        }

        if not is_correct:
            mismatches.append(record)

        if save_all:
            all_results.append(record)

    n_processed = n_correct + n_mismatch
    pct_correct  = 100 * n_correct  / n_processed if n_processed > 0 else 0
    pct_mismatch = 100 * n_mismatch / n_processed if n_processed > 0 else 0

    summary_lines = [
        f"Anàlisi d'alineació CLIP - Split: {split_name}",
        f"Generat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*60}",
        f"",
        f"RESUM GENERAL",
        f"  Mostres totals         : {n_total}",
        f"  Mostres processades    : {n_processed}",
        f"  Sense caption          : {n_no_caption}",
        f"  Errors                 : {errors}",
        f"",
        f"ALINEACIÓ CLIP vs SOLUTION_INDEX",
        f"Correctes (CLIP = solució) : {n_correct:6d}  ({pct_correct:.1f}%)",
        f"Incorrectes (CLIP ≠ solució): {n_mismatch:6d}  ({pct_mismatch:.1f}%)",
        f"",
        f"CLIP SCORES (mitjana de paràgrafs)",
        f"  Score imatge solució:",
    ]

    if clip_scores_sol:
        summary_lines += [
            f"    Min    : {min(clip_scores_sol):.4f}",
            f"    Max    : {max(clip_scores_sol):.4f}",
            f"    Mitjà  : {statistics.mean(clip_scores_sol):.4f}",
            f"    Mediana: {statistics.median(clip_scores_sol):.4f}",
            f"",
            f"  Score imatge predita per CLIP:",
            f"    Min    : {min(clip_scores_pred):.4f}",
            f"    Max    : {max(clip_scores_pred):.4f}",
            f"    Mitjà  : {statistics.mean(clip_scores_pred):.4f}",
            f"    Mediana: {statistics.median(clip_scores_pred):.4f}",
        ]

    if mismatches:
        gaps = [m["score_gap"] for m in mismatches]
        summary_lines += [
            f"",
            f"  Gap (score_predicció - score_solució) als mismatches:",
            f"    Min    : {min(gaps):.4f}",
            f"    Max    : {max(gaps):.4f}",
            f"    Mitjà  : {statistics.mean(gaps):.4f}",
            f"    Mediana: {statistics.median(gaps):.4f}",
            f"",
            f"  Distribució del gap:",
        ]
        for thr in [0.01, 0.02, 0.05, 0.10, 0.20]:
            n_big = sum(1 for g in gaps if g >= thr)
            summary_lines.append(
                f"    gap >= {thr:.2f}: {n_big:6d} mostres ({100*n_big/len(gaps):.1f}%)"
            )

    summary_lines += [
        f"",
        f"INTERPRETACIÓ",
        f"  Un {pct_mismatch:.1f}% de les mostres tenen una caption que",
        f"  CLIP considera més afí a una imatge diferent de la solució.",
        f"  Això suggereix que el text no sempre descriu la imatge",
        f"  'correcta' del dataset, o que la relació és narrativa/contextual",
        f"  en lloc de purament visual.",
    ]

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    return mismatches, all_results, summary_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",     nargs="+",
                        default=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--save_all",   action="store_true",
                        help="Guarda també les mostres correctes")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Dispositiu : {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nCarregant CLIP ({CLIP_MODEL})...")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()
    print("  CLIP carregat ")

    for split in args.splits:
        mismatches, all_results, summary_text = process_split(
            split, clip_model, clip_processor, device,
            args.batch_size, args.save_all
        )

        mismatch_path = os.path.join(args.output_dir, f"mismatches_{split}.json")
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatches, f, ensure_ascii=False, indent=2)
        print(f"\n Mismatches guardats: {mismatch_path} ({len(mismatches)} mostres)")

        summary_path = os.path.join(args.output_dir, f"summary_{split}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f" Resum guardat: {summary_path}")

        if args.save_all and all_results:
            all_path = os.path.join(args.output_dir, f"all_results_{split}.json")
            with open(all_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f" Tots els resultats guardats: {all_path}")

    print(f"\n{'='*60}")
    print(f"  Anàlisi completada. Resultats a: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()