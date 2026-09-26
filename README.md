# Comics Retrieval - Model Multimodal de Recuperació de Vinyetes (Grup 09)

Aquest projecte implementa un sistema de recuperació multimodal (**Image-to-Text / Text-to-Image**) aplicat al món del còmic, desenvolupat com a pràctica de l'assignatura de **Xarxes Neuronals i Aprenentatge Profund (XNAP)**. L'objectiu principal és alinear les representacions semàntiques del text de les descripcions amb les característiques visuals de les vinyetes del dataset **COMIC-PAP**.

La nostra arquitectura final (**Baseline**) combina un extractor de text congelat (**SentenceBERT**) i un extractor de característiques visuals (**ResNet50** amb la capa `layer4` entrenable), optimitzat mitjançant la funció **TripletMarginLoss** (`margin = 0.4`). El dataset s'ha filtrat utilitzant un llindar de similitud de **CLIP = 0.29** per garantir la neteja de les parelles d'entrenament, i el model final s'ha validat sobre un conjunt net de **262 exemples**.

---

## Resultats Destacats

A partir de l'exploració d'hiperparàmetres realitzada amb els Sweeps de **Weights & Biases** (run `firm-sweep-24`), es va llançar un entrenament estès a **150 epochs**. El model va assolir els següents resultats de convergència final al conjunt de validació:

* **Recall@1:** 0.0458
* **Recall@10:** 0.1946
* **MRR (Mean Reciprocal Rank):** 0.0900

---

## Estructura Real del Codi i Repositori

El repositori està organitzat amb els següents fitxers i carpetes del projecte final:

* **final_main.py**: Executa l'entrenament de convergència del model final a 150 epochs.
* **main.py**: Configura i llança l'exploració d'hiperparàmetres (Sweeps) a WandB.
* **train.py**: Conté el bucle principal d'entrenament i validació per èpoques.
* **test.py**: Script per calcular les mètriques globals de validació i executar el mode demo.
* **run_all.py**: Script d'automatització per llançar diferents execucions o pipelines del projecte.
* **compute_clip_scores.py**: Script utilitzat per calcular i extreure les puntuacions de CLIP originals.
* **analyse_captions.py**: Script d'anàlisi exhaustiva del text de les descripcions del dataset.
* **analyse_solution_alignment.py**: Eina d'anàlisi de l'alineament semàntic final aconseguit.

### Carpetes del Projecte

* **dataloaders/**: Conté `dataset.py`, encarregat de la càrrega i del filtratge per llindar CLIP (0.29).
* **models/**: Definició de l'arquitectura de la xarxa a `models.py`.
* **utils/**: Funcions auxiliars i d'instanciació de components (`utils.py`).
* **clip_scores/**: Carpetes on es guarden els fitxers JSON amb els scores de CLIP calculats.
* **alignment_analysis/** i **analisi_output/**: Directoris de sortida de les anàlisis de dades i gràfics.

### Documents

* **analisis.md**: Document amb l'anàlisi detallada de resultats de la run `firm-sweep-24`.
* **test_scrip.md**: Guia d'execució de proves pas a pas i resolució de problemes a la VM de la UAB.
* **conclusions.md**: Document resum amb les conclusions extretes del projecte.
* **arquitectures.md**: Detall i especificacions de les proves realitzades sobre l'arquitectura del model.
* **environment.yml**: Arxiu de configuració de l'entorn virtual de Conda.
* **demo_panel.png**, **image.png**, **image-1.png**: Imatges i gràfics utilitzats a les demostracions i documentació.

---

## Configuració de l'Entorn i Execució

Abans de llançar qualsevol script, cal crear l'entorn virtual de Conda amb les dependències especificades a `environment.yml` i activar-lo:

```bash
conda env create -f environment.yml
conda activate xnap-example
```

O alternativament instal·lar les dependències amb `pip`:

```bash
pip install torch torchvision transformers sentence-transformers datasets wandb pytorch-metric-learning tqdm
```

---

## Entrenament del Model Final

Per reproduir l'entrenament de 150 epochs amb els millors hiperparàmetres trobats:

```bash
python final_main.py
```

Paràmetres utilitzats:

* Batch Size = 64
* Learning Rate = 1e-4
* Margin = 0.4
* Epochs = 150
* CLIP Threshold aplicat: 0.29


---

## Avaluació i Test

Per executar l'avaluació global de mètriques o provar el mode demostració:

```bash
python test.py
```

Per a instruccions avançades, configuració de variables d'entorn i resolució de problemes relacionats amb la VM de la UAB, consulteu:

```text
test_scrip.md
```

---

## Contributors

### Grup 09

* **Marc Dalmau Guamis** 
* **Alejandro Flores Hernández** 
* **Javier Martínez Hernández** 
* **Dídac Tresserres Saló** 

---

## Xarxes Neuronals i Aprenentatge Profund

**Grau en Matemàtica Computacional i Analítica de Dades (MatCAD)**

**Universitat Autònoma de Barcelona (UAB)**

**Curs 2025-2026**
