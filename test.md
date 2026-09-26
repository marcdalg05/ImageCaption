# Guia d'ús — Comics Retrieval (Grup 09)

Aquest document explica pas a pas com executar el projecte: des de la preparació de l'entorn fins al llançament del training, la demo i la sincronització amb Weights & Biases. Està pensat per a algú que no ha treballat mai amb el codi.

---

## Requisits previs

Assegura't de tenir:
- Python 3.10 o superior
- Accés a una GPU (CUDA)
- Accés al dataset COMIC-PAP a `/home/datasets/COMIC-PAP`
- Compte a [wandb.ai](https://wandb.ai) (gratuït)

Instal·la les dependències:

```bash
conda env create -f environment.yml
conda activate xnap-example
```

O amb pip:

```bash
pip install torch torchvision transformers sentence-transformers datasets wandb pytorch-metric-learning tqdm
```

---

## 1. Entrenar el model final (150 epochs, millors hiperparàmetres)

Aquest és el cas principal: llançar el run final amb els hiperparàmetres del millor model trobat al sweep (`firm-sweep-24`).

```bash
python final_main.py
```

El script et demanarà les credencials de WandB la primera vegada. Introdueix la teva API key quan la demani (la trobes a [wandb.ai/settings](https://wandb.ai/settings)).

**Què fa**:
1. Carrega el dataset de training filtrat per CLIP (threshold 0.29)
2. Construeix el model: SentenceBERT (totalment congelat) + ResNet50 (layer4 entrenable)
3. Entrena 150 epochs amb TripletMarginLoss (margin=0.4), lr=1e-4, AdamW weight_decay=0.1
4. Cada epoch calcula R@1, R@10 i MRR al validation set complet
5. Guarda el millor checkpoint a `trained_models/model_best.pth` (millor R@1)
6. Guarda el model final a `trained_models/model_final_150ep.pth`
7. Sincronitza totes les mètriques en temps real a WandB (projecte: `Comics-Retrieval-Final`)

**Temps estimat**: ~6-10 hores depenent de la GPU.

---

## 2. Executar en segon pla (recomanat per a execucions llargues)

Per evitar que el procés mori si tanques la terminal o la sessió SSH:

```bash
nohup python final_main.py > output.log 2>&1 &
```

- `nohup`: el procés no es mata en tancar la terminal
- `> output.log 2>&1`: tot el log (stdout + errors) es guarda a `output.log`
- `&`: el procés s'executa en segon pla (background)

Per veure el progrés en temps real:

```bash
tail -f output.log
```

Per veure si el procés segueix en marxa:

```bash
ps aux | grep final_main.py
```

Per aturar-lo:

```bash
kill <PID>
```

El PID és el número que apareix a la primera columna del resultat de `ps aux`.

---

## 3. Llançar un sweep de cerca d'hiperparàmetres

Si vols continuar explorant nous hiperparàmetres:

```bash
python main.py
```

Això crea un nou sweep Bayesià a WandB i executa fins a 50 runs amb combinacions aleatòries dels hiperparàmetres definits al `sweep_config` dins de `main.py`. Cada run s'atura automàticament si no progressa (Hyperband Early Stopping).

Per executar en segon pla:

```bash
nohup python main.py > sweep.log 2>&1 &
tail -f sweep.log
```

---

## 4. Avaluar el model entrenat

Un cop tens un model guardat a `trained_models/model_best.pth`, pots avaluar-lo al test set:

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --eval
```

**Sortida esperada**:
```
========================================
  Test R@1  : 0.XXXX
  Test R@10 : 0.XXXX
  Test MRR  : 0.XXXX
========================================
```

Per enviar els resultats a WandB:

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --eval \
  --wandb
```

---

## 5. Demo: provar el model amb una imatge i textos concrets

El mode demo permet donar una imatge i una llista de textos candidats, i el model retorna quin text és el més similar a la imatge:

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --image path/a/la/imatge.jpg \
  --texts "A hero fights a villain" "A dog sleeps on the sofa" "Two characters talk"
```

**Sortida esperada**:
```
Imatge: path/a/la/imatge.jpg
──────────────────────────────────────────────────
  [0.8342]  A hero fights a villain
  [0.2103]  A dog sleeps on the sofa
  [0.4871]  Two characters talk
──────────────────────────────────────────────────
  ✓ Millor coincidència: "A hero fights a villain" (sim=0.8342)
```

Els valors entre claudàtors són la similitud cosinus (de -1 a 1). Com més alt sigui el valor, més similar és.

Per combinar la demo i l'avaluació del dataset real de validació (les 262 mostres netes) en la mateixa execució:

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --eval \
  --image panel.jpg \
  --texts "A hero fights" "A dog sleeps"
```

---

## 6. Sincronització amb Weights & Biases

Totes les execucions de `final_main.py` i `main.py` sincronitzen automàticament les mètriques a WandB. Podràs veure en temps real:

- `train/loss`: pèrdua de training per batch
- `train/epoch_loss`: pèrdua mitjana per epoch
- `val/R@1`, `val/R@10`, `val/MRR`: mètriques actuals de cada epoch
- `val/best_R@1`, `val/best_R@10`, `val/best_MRR`: millors valors acumulats

Per accedir al dashboard:
1. Ves a [wandb.ai](https://wandb.ai) i inicia sessió
2. Selecciona el projecte `Comics-Retrieval-Final` (per al run final) o el projecte de sweeps de l'equip per veure les gràfiques de control.
3. Les gràfiques s'actualitzen en temps real mentre s'executa el training

Si WandB no té connexió a internet, el log es guarda localment i es sincronitza quan hi hagi connexió:

```bash
wandb sync wandb/offline-run-XXXXX
```

---

## 7. Estructura de carpetes

```
projecte-matcad_xnap_grup09/
├── final_main.py          # ← Executa el model final (150 epochs)
├── main.py                # ← Sweep de cerca d'hiperparàmetres
├── train.py               # Lògica d'entrenament i validació
├── test.py                # Avaluació i demo
├── models/
│   └── models.py          # Arquitectura del model (Baseline)
├── dataloaders/
│   └── dataset.py         # ComicsPAPDataset amb filtre CLIP
├── utils/
│   └── utils.py           # make(): construeix model, dataloaders, optimizer
├── clip_scores/
│   ├── clip_scores_train.json
│   └── clip_scores_validation.json
├── trained_models/        # (es crea automàticament)
│   ├── model_best.pth     # Millor checkpoint (millor R@1)
│   └── model_final_150ep.pth  # Model al final del run
├── analisis.md
├── arquitectures.md
├── conclusions.md
└── environment.yml
```

---

## 8. Paràmetres del test.py

| Paràmetre | Tipus | Per defecte | Descripció |
|---|---|---|---|
| `--model_path` | string | **obligatori** | Ruta al fitxer `.pth` del model |
| `--embedding_dim` | int | 512 | Dimensió de l'embedding (ha de coincidir amb l'entrenament) |
| `--device` | string | `cpu` | `cpu` o `cuda` |
| `--image` | string | — | [Demo] Ruta a la imatge |
| `--texts` | string+ | — | [Demo] Textos candidats |
| `--eval` | flag | — | [Eval] Avalua sobre el dataset de validació complet |
| `--wandb` | flag | — | [Eval] Envia resultats a WandB |
