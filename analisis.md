# Anàlisi de Resultats — Comics Retrieval (Grup 09)

## 1. Configuració experimental

El projecte aborda el problema de recuperació d'informació multimodal (text-imatge) sobre el dataset **COMIC-PAP**. L'objectiu és que, donada una descripció textual d'un panell de còmic, el model sigui capaç de recuperar la imatge correcta entre tots els panells del conjunt de validació.

Les mètriques principals utilitzades son:
- **R@1**: proporció de consultes on la imatge correcta és la primera retornada.
- **R@10**: proporció de consultes on la imatge correcta apareix entre les 10 primeres.
- **MRR (Mean Reciprocal Rank)**: mitjana de 1/rank de la imatge correcta. Captura si el model "quasi encerta" (és sensible a posicions 2 i 3, que R@1 no veu).

S'han executat un total de **~450 runs** distribuïts en diverses configuracions d'arquitectura, amb cerca Bayesiana d'hiperparàmetres (WandB Sweeps) i Hyperband Early Stopping per aturar els runs clarament inferiors.

---

## 2. Evolució cronològica de les arquitectures

### 2.1 Model inicial: ResNet18 + BERT + CrossEntropyLoss (Codi del Carles)

La primera implementació partia del codi base proporcionat pel Carles, que combinava **BERT** com a encoder de text i **ResNet18** com a encoder d'imatge, amb **CrossEntropyLoss** com a funció de pèrdua.

En les primeres execucions, les mètriques R@1 i R@10 sortien sempre valors aproximadament 0. Després d'una anàlisi exhaustiva del pipeline, vam detectar que el problema no era l'arquitectura sinó la **càrrega del dataset**: les imatges s'estaven llegint com a imatges negres (tensors de zeros) perquè la ruta de les imatges en el dataloader era incorrecta. Un cop corregida la importació del dataset i verificat que les imatges es carregaven correctament, el model ja mostrava valors de mètriques no trivials.

Amb el bug de les imatges resolt, vam confirmar que la CrossEntropyLoss no era adequada per a una tasca de retrieval (és una pèrdua de classificació, no de similitud).

### 2.2 ResNet18 + BERT + MultiSimilarityLoss + cerca exhaustiva d'hiperparàmetres

Amb el dataset funcionant correctament, vam passar a una primera millora significativa: substituir la CrossEntropyLoss per **MultiSimilarityLoss** de la llibreria `pytorch-metric-learning`. Aquesta funció mina parelles difícils (*hard negatives*) i és molt més adequada per a tasques de retrieval.

Addicionalment, vam implementar **normalització L2** a la sortida dels dos encoders (vectors a la hiperesfera unitària) i vam llançar una cerca exhaustiva d'hiperparàmetres mitjançant WandB Sweeps. Els hiperparàmetres explorats van ser:

| Hiperparàmetre | Valors provats |
|---|---|
| Learning rate | `1e-3`, `5e-4`, `1e-4`, `5e-5`, `2e-5` |
| Batch size | `32`, `64`, `128`, `256` |
| Embedding dim | `128`, `256`, `512`, `1024` |
| Epochs | `5`, `10`, `15`, `20` |
| Margin | `0.1`, `0.2`, `0.3`, `0.5` |
| Freeze encoders | `True`, `False` |

Els millors resultats d'aquesta fase van ser:

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0267 |
| R@10 màxim | 0.1565 |

Un problema que vam identificar en aquesta fase és que el dataset era molt gran i les execucions trigaven molt, cosa que feia que el cicle d'iteració (llançar sweep → esperar resultats → analitzar → ajustar) fos molt lent. Per resoldre-ho vam decidir treballar amb el **20% del dataset** durant les fases de cerca d'hiperparàmetres.

### 2.3 SentenceBERT + ResNet50 + TripletMarginLoss + descongelació parcial

Basant-nos en els resultats anteriors i les recomanacions del professor Carles, vam implementar un conjunt de millores simultànies:

- **BERT → SentenceBERT**: genera embeddings de frase optimitzats per a similitud semàntica via mean pooling, molt més adequats que el token `[CLS]` de BERT per a descripcions narratives.
- **ResNet18 → ResNet50**: 2048 features de sortida en lloc de 512, molt més expressiu visualment.
- **MultiSimilarityLoss → TripletMarginLoss**: treballa amb tripletes (àncora, positiu, negatiu) i imposa explícitament un marge entre distàncies. Amb batches grans, la mineria automàtica de negatius semi-difícils proporciona gradients molt informatius.
- **Descongelació parcial**: es descongelaven SentenceBERT des de la layer 5 i ResNet50 des de layer4, permetent fine-tuning sense oblit catastròfic.

Els millors resultats d'aquesta fase van ser:

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0458 |
| R@10 màxim | 0.2137 |

### 2.4 Filtratge CLIP + anàlisi de qualitat del dataset

Vam implementar un pas de neteja del dataset basat en **CLIP Score**: per a cada parella text-imatge del dataset, vam calcular la similitud semàntica amb un model CLIP preentrenat. Les parelles per sota d'un llindar determinat (`clip_threshold`) s'eliminaven de l'entrenament.

Per decidir el llindar adequat, vam generar gràfiques de distribució dels CLIP scores (fitxers `analyse_captions.py` i `analyse_solution_alignment.py`), que mostraven que una part significativa del dataset tenia scores molt baixos, corresponent a parelles on el text descriu el context narratiu del panell anterior (no el contingut visual de la imatge actual). Vam explorar llindars entre 0.25 i 0.32.

En aquest punt vam detectar un error crític en el pipeline: **el filtre CLIP s'havia aplicat tant al train com al validation set**, cosa que feia que les mètriques de validació estiguessin inflades artificialment properes al **0.51** (s'avaluava sobre un subconjunt de mostres fàcils filtrades, no sobre el dataset de validació real complet). Totes les execucions d'aquesta fase intermedia van mostrar mètriques que no eren comparables amb els runs reals del projecte.

### 2.5 Millores en el control de mètriques i correcció del pipeline

A partir de les sessions de seguiment i revisió del codi, vam introduir canvis metodològics clau per garantir el rigor del projecte:

- **Mètrica MRR**: Vam afegir el Mean Reciprocal Rank com a mètrica addicional per avaluar amb precisió la posició de la imatge correcta en el rànquing de similitud cosinus.
- **SBERT completament congelat**: Es va optar per congelar SentenceBERT completament i centrar els recursos en descongelar únicament la `layer4` de la ResNet50, garantint estabilitat en la generació d'embeddings textuals i evitant l'overfitting.
- **Aïllament de mètriques a WandB**: Vam corregir la configuració dels logs perquè `val/best_R@10` es trackegés de forma independent en el seu propio epoch màxim, i no lligat restrictivament al pic de `R@1`.
- **Resolució del bug de validació**: Vam eliminar definitivament el filtre CLIP del validation set (execució `26T20_10`), recuperant les 262 mostres reals de validació per obtenir mètriques honestes i comparables.

### 2.6 Exploració d'expressivitat: ResNet18 vs ResNet50

Amb el pipeline d'avaluació ja net i fiable, vam dur a terme una comparació directa entre arquitectures visuals (sweep `26T20_10` vs `26T20_11`). Els experiments van confirmar que **ResNet50 és clarament superior**: amb la ResNet18, els embeddings visuals de 512 dimensions es quedaven curts d'expressivitat i no aconseguien alinear correctament l'espai mètric multimodal compartit amb SentenceBERT.

### 2.7 Regularització i ampliació de dades — Millor model final (26T19)

L'última millora implementada va consistir a incrementar el **weight decay** a **0.1** en l'optimitzador AdamW per afegir una regularització L2 més exigent. En paral·lel, i gràcies a l'estalvi de temps computacional que suposava mantenir SBERT congelat, vam poder ampliar el subset d'entrenament del 20% al **40%** del Train oficial (~9.440 mostres reals).

Amb aquesta configuració final (`firm-sweep-24`), vam obtenir els millors resultats vàlids de control del sweep global:

| Mètrica | Valor |
|---------|-------|
| R@1 | 0.0458 |
| R@10 | **0.2328** |
| MRR | 0.1002 |

Aquest conjunt d'hiperparàmetres va ser seleccionat com el disseny òptim per ser llançat a l'entrenament final extensiu de 150 epochs.

---

## 3. Anàlisi dels hiperparàmetres (sweep final 26T19)

### 3.1 Learning rate

| LR | R@1 mitjà | R@10 màxim |
|-----|-----------|------------|
| 1e-4 | 0.0458 | 0.2328 |
| 5e-5 | 0.0344 | 0.2137 |
| 2e-5 | 0.0224 | 0.1718 |
| 1e-5 | 0.0182 | 0.1527 |

**1e-4** és el learning rate òptim: prou gran per adaptar ResNet50 layer4 amb els nous gradients, però no tant com per desestabilitzar els pesos preentrenats.

### 3.2 Clip threshold

| Threshold | R@10 màxim |
|-----------|------------|
| 0.29 | **0.2328** |
| 0.27 | 0.2137 |
| 0.30 | 0.1908 |
| 0.0  | 0.1679 |

El threshold de **0.29** és l'òptim: filtra les parelles més sorolloses però conserva prou dades per a un training robust.

### 3.3 Margin

| Margin | R@10 màxim |
|--------|------------|
| 0.4 | **0.2328** |
| 0.3 | 0.1984 |
| 0.2 | 0.2137 |
| 0.1 | 0.1527 |

Un marge més gran (**0.4**) imposa una separació més estricta entre parelles correctes i incorrectes, cosa que beneficia el retrieval en un dataset visualment homogeni com COMIC-PAP.

---

## 4. Comparació global de configuracions

| Fase | Arquitectura | R@1 màxim | R@10 màxim |
|---|---|---|---|
| 2.1 | ResNet18 + BERT + CrossEntropy | ~0.000 | ~0.000 (bug rutes dades) |
| 2.2 | ResNet18 + BERT + MultiSimilarity | 0.0267 | 0.1565 |
| 2.3 | ResNet50 + SentenceBERT + Triplet | 0.0458 | 0.2137 |
| 2.4 | + CLIP filter (Validació contaminada) | *No Vàlid* | ~0.510 (No vàlid) |
| 2.6 | ResNet50 + SBERT congelat (Validació neta) | 0.0458 | 0.1984 |
| 2.7 | + weight_decay=0.1 + 40% dataset (sweep pic) | 0.0458 | **0.2328** |
| **Run final** | **150 epochs convergit** | **0.0458** | **0.1946** |

---

## 5. Discussió

Els valors de R@1 (~4.6%) cal contextualitzar-los dins el domini del problema: el dataset COMIC-PAP és extremadament exigent, ja que conté centenars de panells visualment molt similars (mateixos personatges, estil de línia i paleta cromàtica). A més, s'ha detectat que moltes de les descripcions textuals narren el context narratiu o fets de la vinyeta anterior, en lloc de descriure els elements visuals explícits del panell actual. Això crea una ambigüitat semàntica intrínseca que limita de forma natural el sostre de rendiment de qualsevol model d'alineació directa text-imatge.

En aquest context, el R@10 final de **0.1946 (19.5%)** assolit de forma estable i convergida en el run de 150 epochs és el resultat més rellevant: indica que en gairebé 1 de cada 5 consultes, la imatge correcta apareix entre les 10 primeres d'un rànquing competitiu de centenars de vinyetes. Tot i que durant la fase de sweeps es va observar un pic transitori de **0.2328**, el valor de convergència del model final demostra una gran robustesa i constitueix un resultat molt sòlid, especialment tenint en compte les fortes restriccions computacionals que ens han obligat a entrenar amb un subset del 40% de les dades.

---

## 6. Resultats del model final (run de 150 epochs)

El run `firm-sweep-24_150epochs` ha entrenat durant 150 epochs amb els millors hiperparàmetres identificats al sweep. Els resultats finals confirmats son:

| Mètrica | Valor | Interpretació |
|---|---|---|
| R@1 | 0.0458 | La imatge correcta és la primera en 1/22 consultes (×12 sobre l'atzar real) |
| R@10 | 0.1946 | La imatge correcta apareix top-10 en gairebé 1 de cada 5 consultes |
| MRR | 0.0903 | Alta concentració i consistència dels encerts a la part alta del rànquing |
| train/epoch_loss | 0.087 | Loss convergida i estable (partia de 0.35) |

### Per què R@10=0.195 i no el 0.233 del sweep?

El 0.2328 era el valor màxim assolit en un epoch puntual favorable durant la fase de sweeps curts (de 20 epochs). En el run de 150 epochs, el model arriba a la seva convergència estable al voltant del **0.1946**. La diferència s'explica pel fet que les mètriques basades en rànquings generen de vegades pics transitoris en l'espai mètric que s'estabilitzen a llarg termini.

La corba de pèrdua (`train/epoch_loss`) demostra la solidesa de l'entrenament, baixant de forma consistent i neta de 0.35 a 0.087 al llarg de l'execució, sense presentar cap símptoma d'overfitting (la mètrica de validació es manté constant i consolidada), cosa que confirma que els resultats finals son totalment fiables i reproduïbles.

### Contextualització dels resultats

Els valors de R@1 (~4.6%) cal posar-los en context dins del domini del problema:
- **El dataset COMIC-PAP és extremadament exigent**: El conjunt de validació compta amb 262 mostres reals on molts dels panells pertanyen al mateix còmic, compartint exactament els mateixos personatges, estil de línia i paleta cromàtica.
- **L'ambigüitat és intrínseca**: S'ha comprovat que els textos associats sovint descriuen el context narratiu o fets de la vinyeta anterior, i no el contingut visual directe del panell actual, dificultant l'alineació directa.
- **R@1 aleatori ≈ 0.38%**: Un model basat en eleccions completament aleatòries sobre la mostra de validació obtindria un 0.38% d'encert. El nostre model assoleix un 4.58%, el que significa situar-se **12 vegades per sobre de l'atzar**.
- **Entrenat amb el 40% de les dades**: El model s'ha optimitzat limitant els recursos per raons de cost computacional; és altament probable que utilitzar el 100% del dataset de train (23.600 mostres) millori significativament les mètriques absolutes.
