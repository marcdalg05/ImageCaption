
import io
import re
import sys
import json
import csv
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

print("Carregant llibreries...", flush=True)

from datasets import load_dataset
print("datasets", flush=True)

from transformers import CLIPProcessor, CLIPModel
print("transformers (CLIP)", flush=True)

from sentence_transformers import SentenceTransformer
print("sentence_transformers", flush=True)

from sklearn.metrics.pairwise import cosine_similarity
print("sklearn", flush=True)

from PIL import Image
print("PIL", flush=True)

print("Totes les llibreries carregades!\n", flush=True)


DATASET_PATH    = "/home/datasets/COMIC-PAP/datasets/VLR-CVC___comics_pap"
DATASET_NAME    = "default"
CLIP_THRESHOLD  = 0.22
SBERT_THRESHOLD = 0.85
ALL_SPLITS      = ["train", "validation", "test"]
MAX_POS         = 8

COLORS = {
    "train":      "#4C72B0",
    "validation": "#DD8452",
    "test":       "#55A868",
    "global":     "#8172B2",
}


def setup_logger(out_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("analyse_captions")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "analyse.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger



def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}|\n', text) if p.strip()]


def get_correct_image(sample) -> Image.Image:
    sol_idx = max(sample["solution_index"], 0)
    img = sample["options"][sol_idx]
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if not isinstance(img, Image.Image):
        return Image.fromarray(img).convert("RGB")
    return img


@torch.no_grad()
def compute_clip_scores(clip_model, clip_processor, device,
                        image: Image.Image, paragraphs: list[str]) -> list[float]:
    truncated = [p[:300] for p in paragraphs]
    inputs = clip_processor(
        text=truncated,
        images=[image] * len(truncated),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    outputs  = clip_model(**inputs)
    img_norm = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return (img_norm * txt_norm).sum(dim=-1).cpu().numpy().tolist()


def compute_sbert_redundancy(sbert_model, paragraphs: list[str],
                              clip_scores_list: list[float]) -> tuple[list[float], int]:
    if len(paragraphs) == 1:
        return [0.0], 0
    ref_idx    = int(np.argmax(clip_scores_list))
    embeddings = sbert_model.encode(paragraphs, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    redundancy = []
    for i in range(len(paragraphs)):
        sims = [sim_matrix[i][j] for j in range(len(paragraphs)) if j != i]
        redundancy.append(float(max(sims)) if sims else 0.0)
    return redundancy, ref_idx


def filter_paragraphs(c_scores: list[float], s_scores: list[float],
                      ref_idx: int, clip_thr: float,
                      sbert_thr: float) -> list[int]:
    keep = [
        j for j, (c, s) in enumerate(zip(c_scores, s_scores))
        if j == ref_idx or (c >= clip_thr and s <= sbert_thr)
    ]
    return keep if keep else [ref_idx]


def nou_acumulador(splits: list[str]) -> dict:
    return {
        "mostres":         0,
        "total_par":       0,
        "total_conservar": 0,
        "total_eliminar":  0,
        "only_ref":        0,
        "ref_positions":   [0] * MAX_POS,
        "clip_per_pos":    [[] for _ in range(MAX_POS)],
        "sbert_per_pos":   [[] for _ in range(MAX_POS)],
        "total_per_pos":   [0] * MAX_POS,
        "elim_per_pos":    [0] * MAX_POS,
        "split_stats": {
            s: {"mostres": 0, "par": 0, "conservar": 0, "eliminar": 0}
            for s in splits
        },
    }


def processar_split(split_name: str, split_data,
                    clip_model, clip_processor, sbert_model, device,
                    acum: dict, clip_thr: float, sbert_thr: float,
                    n_mostres: int | None,
                    logger: logging.Logger) -> list[dict]:
    total    = min(n_mostres, len(split_data)) if n_mostres else len(split_data)
    iterador = split_data.select(range(total)) if n_mostres else split_data
    errors   = 0
    eliminats_split: list[dict] = []

    logger.info(f"Processant {total} mostres del split '{split_name}'...")

    for i, sample in enumerate(tqdm(iterador, desc=f"  {split_name}", total=total)):
        if i % 50 == 0 and i > 0:
            logger.debug(f"[{i}/{total}] {acum['total_eliminar']} paràgrafs "
                         f"candidats a eliminar fins ara")
        try:
            image      = get_correct_image(sample)
            caption    = sample.get("previous_panel_caption", "") or ""
            paragraphs = split_paragraphs(caption)
            if not paragraphs:
                continue

            c_scores          = compute_clip_scores(clip_model, clip_processor,
                                                    device, image, paragraphs)
            s_scores, ref_idx = compute_sbert_redundancy(sbert_model,
                                                         paragraphs, c_scores)
            keep              = filter_paragraphs(c_scores, s_scores,
                                                  ref_idx, clip_thr, sbert_thr)
            keep_set          = set(keep)
            n_elim            = len(paragraphs) - len(keep)

            acum["mostres"]         += 1
            acum["total_par"]       += len(paragraphs)
            acum["total_conservar"] += len(keep)
            acum["total_eliminar"]  += n_elim
            st = acum["split_stats"][split_name]
            st["mostres"]   += 1
            st["par"]       += len(paragraphs)
            st["conservar"] += len(keep)
            st["eliminar"]  += n_elim

            if len(keep) == 1:
                acum["only_ref"] += 1
            acum["ref_positions"][min(ref_idx, MAX_POS - 1)] += 1

            for pos, (c, s) in enumerate(zip(c_scores[:MAX_POS], s_scores[:MAX_POS])):
                acum["clip_per_pos"][pos].append(c)
                acum["total_per_pos"][pos] += 1
                if pos not in keep_set:
                    acum["elim_per_pos"][pos] += 1
                if pos != ref_idx:
                    acum["sbert_per_pos"][pos].append(s)

            if n_elim > 0:
                removed = []
                for j in range(len(paragraphs)):
                    if j not in keep_set:
                        motius = []
                        if c_scores[j] < clip_thr:
                            motius.append(f"CLIP={c_scores[j]:.4f}<{clip_thr}")
                        if s_scores[j] > sbert_thr:
                            motius.append(f"SBERT={s_scores[j]:.4f}>{sbert_thr}")
                        removed.append({
                            "paragraph_index": j,
                            "text":            paragraphs[j],
                            "clip_score":      round(c_scores[j], 4),
                            "sbert_score":     round(s_scores[j], 4),
                            "reason":          " | ".join(motius),
                        })
                eliminats_split.append({
                    "sample_id":          i,
                    "split":              split_name,
                    "total_paragraphs":   len(paragraphs),
                    "kept_paragraphs":    len(keep),
                    "removed_paragraphs": removed,
                })

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Error a mostra {i}: {e}")
            elif errors == 6:
                logger.warning("(s'ometen els errors següents per brevetat...)")

    if errors:
        logger.warning(f"{errors} errors ignorats en el split '{split_name}'.")

    return eliminats_split


def desar_eliminats(tots_eliminats: list[dict], out_dir: Path,
                    clip_thr: float, sbert_thr: float) -> None:
    """Guarda els paràgrafs a eliminar en JSON i CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = {
        "generated_at":               ts,
        "clip_threshold":             clip_thr,
        "sbert_threshold":            sbert_thr,
        "total_samples_with_removals": len(tots_eliminats),
        "total_paragraphs_to_remove":  sum(
            len(e["removed_paragraphs"]) for e in tots_eliminats
        ),
        "data": tots_eliminats,
    }
    json_path = out_dir / "paragraphs_to_remove.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"JSON guardat: {json_path}")

    csv_path = out_dir / "paragraphs_to_remove.csv"
    fieldnames = ["split", "sample_id", "total_paragraphs", "kept_paragraphs",
                  "paragraph_index", "clip_score", "sbert_score", "reason", "text"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in tots_eliminats:
            for p in entry["removed_paragraphs"]:
                writer.writerow({
                    "split":            entry["split"],
                    "sample_id":        entry["sample_id"],
                    "total_paragraphs": entry["total_paragraphs"],
                    "kept_paragraphs":  entry["kept_paragraphs"],
                    "paragraph_index":  p["paragraph_index"],
                    "clip_score":       p["clip_score"],
                    "sbert_score":      p["sbert_score"],
                    "reason":           p["reason"],
                    "text":             p["text"].replace("\n", " "),
                })
    print(f"CSV  guardat: {csv_path}")

def _posicions_valides(data_per_pos: list[list]) -> tuple[list[int], list[float], list[float]]:
    positions, means, stds = [], [], []
    for pos, vals in enumerate(data_per_pos):
        if vals:
            positions.append(pos)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    return positions, means, stds


def generar_grafic_split(split_name: str, acum: dict,
                         clip_thr: float, sbert_thr: float,
                         out_dir: Path) -> None:
    color = COLORS.get(split_name, "#333333")
    fig   = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Anàlisi de captions — Split: {split_name.upper()}\n"
        f"CLIP thr={clip_thr}  |  SBERT thr={sbert_thr}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    st         = acum["split_stats"].get(split_name, {})
    n_mostres  = st.get("mostres", 0)
    n_par      = st.get("par", 0)
    n_conserva = st.get("conservar", 0)
    n_elim     = st.get("eliminar", 0)

    ax1 = fig.add_subplot(gs[0, 0])
    positions, means, stds = _posicions_valides(acum["clip_per_pos"])
    if positions:
        bars = ax1.bar([f"P{p}" for p in positions], means,
                       yerr=stds, color=color, alpha=0.8,
                       error_kw={"elinewidth": 1.5, "capsize": 4})
        ax1.axhline(clip_thr, color="red", linestyle="--", linewidth=1.5,
                    label=f"Threshold = {clip_thr}")
        for bar, m in zip(bars, means):
            if m < clip_thr:
                bar.set_color("#E74C3C")
                bar.set_alpha(0.7)
        ax1.set_title("CLIP Score mitjà per posició", fontsize=11)
        ax1.set_xlabel("Posició del paràgraf")
        ax1.set_ylabel("CLIP Score (cosinus)")
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, max(max(means) * 1.3, clip_thr * 1.5) if means else 0.5)
        for p, m in zip([f"P{p}" for p in positions], means):
            ax1.text(p, m + max(means) * 0.02, f"{m:.3f}",
                     ha="center", va="bottom", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    positions_s, means_s, stds_s = _posicions_valides(acum["sbert_per_pos"])
    if positions_s:
        bars2 = ax2.bar([f"P{p}" for p in positions_s], means_s,
                        yerr=stds_s, color=color, alpha=0.8,
                        error_kw={"elinewidth": 1.5, "capsize": 4})
        ax2.axhline(sbert_thr, color="red", linestyle="--", linewidth=1.5,
                    label=f"Threshold = {sbert_thr}")
        for bar, m in zip(bars2, means_s):
            if m > sbert_thr:
                bar.set_color("#E74C3C")
                bar.set_alpha(0.7)
        ax2.set_title("Redundància SBERT per posició\n(màxim cosinus amb qualsevol altre P)",
                      fontsize=11)
        ax2.set_xlabel("Posició del paràgraf")
        ax2.set_ylabel("Similitud cosinus màxima")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.1)
        for p, m in zip([f"P{p}" for p in positions_s], means_s):
            ax2.text(p, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    pos_elim, pcts, labels_elim = [], [], []
    for pos in range(MAX_POS):
        tot = acum["total_per_pos"][pos]
        eli = acum["elim_per_pos"][pos]
        if tot > 0:
            pos_elim.append(f"P{pos}")
            pcts.append(100 * eli / tot)
            labels_elim.append(f"{eli}/{tot}")
    if pos_elim:
        ax3.bar(pos_elim, pcts, color=color, alpha=0.8)
        ax3.axhline(15, color="green",  linestyle=":", linewidth=1.2, label="Ref. 15% (mínim ideal)")
        ax3.axhline(30, color="orange", linestyle=":", linewidth=1.2, label="Ref. 30% (màxim ideal)")
        ax3.axhline(40, color="red",    linestyle="--", linewidth=1.2, label="40% (massa agressiu)")
        ax3.set_title("% paràgrafs eliminats per posició", fontsize=11)
        ax3.set_xlabel("Posició del paràgraf")
        ax3.set_ylabel("% eliminats")
        ax3.set_ylim(0, 110)
        ax3.legend(fontsize=8)
        for pos, pct, lab in zip(pos_elim, pcts, labels_elim):
            ax3.text(pos, pct + 2, f"{pct:.1f}%\n({lab})",
                     ha="center", va="bottom", fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    pct_elim = 100 * n_elim     / n_par      if n_par      else 0
    pct_cons = 100 * n_conserva / n_par      if n_par      else 0
    only     = acum["only_ref"]
    pct_only = 100 * only / n_mostres if n_mostres else 0

    if pct_elim > 40:
        estat = "Threshold massa AGRESSIU"
    elif pct_elim < 10:
        estat = "Threshold massa PERMISSIU"
    else:
        estat = "Threshold en rang ideal"

    resum_lines = [
        f"Split:           {split_name}",
        f"Mostres:         {n_mostres:,}",
        f"Total paràgrafs: {n_par:,}",
        "",
        f"Conservats:   {n_conserva:,}  ({pct_cons:.1f}%)",
        f"Eliminats:   {n_elim:,}  ({pct_elim:.1f}%)",
        "",
        f"Mostres amb sols 1 paràgraf: {only} ({pct_only:.1f}%)",
        "",
        f"Avaluació: {estat}",
        "",
        "Thresholds aplicats:",
        f"  CLIP  < {clip_thr}  (poc contingut visual)",
        f"  SBERT > {sbert_thr}  (redundant semànticament)",
        "",
        "Interpretació:",
        "  >40% eliminat → threshold massa agressiu",
        "  <10% eliminat → threshold massa permissiu",
        "  Ideal: 15-30% per millorar el Recall",
    ]
    ax4.text(0.05, 0.95, "\n".join(resum_lines),
             transform=ax4.transAxes,
             fontsize=10, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FF",
                       edgecolor="#AABBDD", alpha=0.9))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grafiques_{split_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gràfica guardada: {out_path}")


def generar_grafic_global(splits: list[str], acum: dict,
                          clip_thr: float, sbert_thr: float,
                          out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Comparativa entre splits  —  CLIP thr={clip_thr} | SBERT thr={sbert_thr}",
        fontsize=13, fontweight="bold",
    )
    posicions_labels = [f"P{p}" for p in range(MAX_POS)]

    for ax, (title, key, thr) in zip(
        axes,
        [
            ("CLIP Score mitjà per posició",     "clip_per_pos",  clip_thr),
            ("SBERT Redundància per posició",     "sbert_per_pos", sbert_thr),
            ("% paràgrafs eliminats per posició", None,            None),
        ]
    ):
        for split_name in splits:
            color = COLORS.get(split_name, "#666666")
            st    = acum["split_stats"].get(split_name, {})
            if st.get("mostres", 0) == 0:
                continue

            if key is not None:
                vals  = acum[key]
                means = [np.mean(v) if v else np.nan for v in vals[:MAX_POS]]
                ax.plot(posicions_labels, means, marker="o", label=split_name,
                        color=color, linewidth=2, markersize=6)
            else:
                pcts = []
                for pos in range(MAX_POS):
                    tot = acum["total_per_pos"][pos]
                    eli = acum["elim_per_pos"][pos]
                    pcts.append(100 * eli / tot if tot else np.nan)
                ax.plot(posicions_labels, pcts, marker="s", label=split_name,
                        color=color, linewidth=2, markersize=6)

        if thr is not None:
            ax.axhline(thr, color="red", linestyle="--", linewidth=1.2,
                       label=f"Threshold {thr}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Posició del paràgraf")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("CLIP Score")
    axes[1].set_ylabel("Similitud SBERT màxima")
    axes[2].set_ylabel("% eliminats")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grafiques_global.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gràfica global guardada: {out_path}")

def barra_ascii(valor: float, maxim: float, amplada: int = 30, char: str = "█") -> str:
    n = int(round(valor / maxim * amplada)) if maxim > 0 else 0
    return char * n + "░" * (amplada - n)


def imprimir_resum_terminal(acum: dict, splits: list[str],
                             clip_thr: float, sbert_thr: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RESUM DE L'ANÀLISI DE CAPTIONS")
    print(f"  Thresholds: CLIP < {clip_thr}  |  SBERT > {sbert_thr}")
    print(sep)

    for s in splits:
        st = acum["split_stats"][s]
        if st["mostres"] == 0:
            continue
        pct_c = 100 * st["conservar"] / st["par"] if st["par"] else 0
        pct_e = 100 * st["eliminar"]  / st["par"] if st["par"] else 0
        print(f"  {s:12s}: {st['mostres']:5d} mostres | {st['par']:6d} paràgrafs "
              f"→ conservar {st['conservar']:6d} ({pct_c:.1f}%) "
              f"| eliminar {st['eliminar']:5d} ({pct_e:.1f}%)")

    print("  " + "─" * 56)
    tot   = acum["total_par"]
    pct_c = 100 * acum["total_conservar"] / tot if tot else 0
    pct_e = 100 * acum["total_eliminar"]  / tot if tot else 0
    print(f"  {'TOTAL':12s}: {acum['mostres']:5d} mostres | {tot:6d} paràgrafs "
          f"→ conservar {acum['total_conservar']:6d} ({pct_c:.1f}%) "
          f"| eliminar {acum['total_eliminar']:5d} ({pct_e:.1f}%)")
    print(sep)

    print("\n  Paràgraf de referència (CLIP màxim) més freqüent:")
    for pos in range(MAX_POS):
        count = acum["ref_positions"][pos]
        if count == 0:
            continue
        pct   = 100 * count / acum["mostres"]
        barra = barra_ascii(pct, 100, amplada=20)
        print(f"    P{pos}: │{barra}│ {count:5d} vegades ({pct:.1f}%)")

    only = acum["only_ref"]
    pct  = 100 * only / acum["mostres"] if acum["mostres"] else 0
    print(f"\n  Mostres on quedaria 1 sol paràgraf: {only} ({pct:.1f}%)")

    print(f"\n{'─'*60}")
    print("  CLIP Score mitjà per posició:")
    for pos, scores in enumerate(acum["clip_per_pos"]):
        if not scores:
            continue
        m, s = np.mean(scores), np.std(scores)
        flag = " ← CLIP BAIX" if m < clip_thr else ""
        print(f"  P{pos} (n={len(scores):5d}) │{barra_ascii(m, 0.40)}│ {m:.3f} ±{s:.3f}{flag}")

    print(f"\n{'─'*60}")
    print("  SBERT Redundància per posició:")
    for pos, scores in enumerate(acum["sbert_per_pos"]):
        if not scores:
            continue
        m, s = np.mean(scores), np.std(scores)
        flag = " ← REDUNDANT" if m > sbert_thr else ""
        print(f"  P{pos} (n={len(scores):5d}) │{barra_ascii(m, 1.00)}│ {m:.3f} ±{s:.3f}{flag}")

    print(f"\n{'─'*60}")
    print("  % paràgrafs eliminats per posició:")
    for pos in range(MAX_POS):
        tot  = acum["total_per_pos"][pos]
        elim = acum["elim_per_pos"][pos]
        if tot == 0:
            continue
        pct = elim / tot
        print(f"  P{pos} (n={tot:5d}) │{barra_ascii(pct, 1.0)}│ {100*pct:.1f}% ({elim}/{tot})")

    print(f"\n{sep}")
    print("  Interpretació ràpida:")
    print("  - >40% eliminat → threshold massa agressiu")
    print("  - <10% eliminat → threshold massa permissiu")
    print("  - Ideal: 15-30% eliminat per millorar el Recall")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Anàlisi CLIP+SBERT de captions — fusió didac + analyse (v2)"
    )
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--splits", nargs="+", default=ALL_SPLITS,
                        choices=["train", "validation", "val", "test"])
    parser.add_argument("--n_mostres", type=int, default=None,
                        help="Mostres per split (default: totes)")
    parser.add_argument("--clip_thr",  type=float, default=CLIP_THRESHOLD)
    parser.add_argument("--sbert_thr", type=float, default=SBERT_THRESHOLD)
    parser.add_argument("--device",    type=str,   default=None)
    parser.add_argument("--output_dir", type=str,  default="analisi_output",
                        help="Directori arrel on guardar els resultats")
    parser.add_argument("--dry_run", action="store_true",
                        help="Només mostra el resum ASCII, sense guardar res a disc")
    args = parser.parse_args()

    args.splits = ["validation" if s == "val" else s for s in args.splits]

    out_root  = Path(args.output_dir)
    plots_dir = out_root / "plots"
    elim_dir  = out_root / "eliminats"
    log_dir   = None if args.dry_run else out_root

    logger = setup_logger(log_dir)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Dispositiu: {device}")
    if not args.dry_run:
        logger.info(f"Resultats a: {out_root.resolve()}")
    else:
        logger.info("Mode dry_run: no es guardarà res a disc.")

    print("\n[1/2] Carregant CLIP (openai/clip-vit-base-patch32)...", flush=True)
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("CLIP carregat\n", flush=True)

    print("[2/2] Carregant SBERT (all-MiniLM-L6-v2)...", flush=True)
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    print("SBERT carregat\n", flush=True)

    acum           = nou_acumulador(args.splits)
    tots_eliminats = []

    for split_name in args.splits:
        print(f"\n{'═'*60}", flush=True)
        print(f"Carregant split '{split_name}'...", flush=True)
        try:
            split_data = load_dataset(
                args.dataset_path, args.dataset_name, split=split_name
            )
        except Exception as e:
            logger.error(f"Error carregant '{split_name}': {e}")
            logger.error("Comprova el path i el nom de la configuració.")
            continue

        print(f"{len(split_data)} mostres carregades", flush=True)

        eliminats_split = processar_split(
            split_name, split_data,
            clip_model, clip_processor, sbert_model, device,
            acum, args.clip_thr, args.sbert_thr, args.n_mostres, logger,
        )
        tots_eliminats.extend(eliminats_split)

        if not args.dry_run:
            print(f"\nGenerant gràfica per al split '{split_name}'...", flush=True)
            generar_grafic_split(split_name, acum,
                                 args.clip_thr, args.sbert_thr, plots_dir)

    if not args.dry_run and len(args.splits) > 1:
        print(f"\nGenerant gràfica comparativa global...", flush=True)
        generar_grafic_global(args.splits, acum,
                              args.clip_thr, args.sbert_thr, plots_dir)

    if not args.dry_run:
        print(f"\nDesant fitxers d'eliminats...", flush=True)
        desar_eliminats(tots_eliminats, elim_dir, args.clip_thr, args.sbert_thr)

    imprimir_resum_terminal(acum, args.splits, args.clip_thr, args.sbert_thr)

    if not args.dry_run:
        print(f"\nTot completat. Resultats a: {out_root.resolve()}")
        print(f"   {plots_dir}/grafiques_<split>.png   → gràfiques matplotlib")
        print(f"   {elim_dir}/paragraphs_to_remove.json → dades per netejar el dataset")
        print(f"   {elim_dir}/paragraphs_to_remove.csv  → versió llegible")
        print(f"   {out_root}/analyse.log               → log complet\n")
    else:
        print("\nAnàlisi completada (dry_run: cap fitxer guardat).\n")


if __name__ == "__main__":
    main()