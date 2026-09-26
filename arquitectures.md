# Arquitectura del Model Final — Comics Retrieval (Grup 09)

## 1. Visió general

El model final és un **bi-encoder cross-modal** dissenyat per a la tasca de text-to-image retrieval sobre el dataset COMIC-PAP. Donada una descripció textual d'un panell de còmic, el model ha d'identificar la imatge corresponent entre tots els panells del conjunt de validació.

L'arquitectura projecta tant el text com la imatge a un **espai d'embedding compartit** de 512 dimensions, on la similitud cosinus entre vectors indica la probabilitat de que una parella text-imatge sigui correcta. Durant la inferència, per a cada consulta textual es calcula la similitud amb tots els embeddings d'imatge del conjunt de validació i es retorna el rànquing resultant.

---

## 2. Components de l'arquitectura

### 2.1 Encoder de text: SentenceBERT (congelat)

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

SentenceBERT és un model basat en Transformer (6 capes, 12 caps d'atenció, dimensió oculta 384) preentrenat específicament per a tasques de similitud semàntica de frases mitjançant *contrastive learning* amb parelles de frases similars i dissimilars.

A diferència de BERT base, que requereix el token `[CLS]` com a representació global, SentenceBERT aplica **mean pooling**: calcula la mitjana ponderada de tots els tokens de sortida (ponderada per l'`attention_mask` per ignorar els tokens de padding), produint un vector de 384 dimensions que encapsula la semàntica completa de la frase.

**Estratègia de congelació**: SentenceBERT es deixa **totalment congelat** durant tot l'entrenament (no es modifiquen els seus pesos). Aquesta decisió la vam prendre després d'analitzar a fons les descripcions de text del dataset. Ens vam adonar que molts textos estaven "desalineats", és a dir, contaven coses genèriques o parlaven del que passava a la vinyeta anterior, en lloc de descriure la imatge actual. Com que el text té aquest límit de qualitat tan clar d'origen, intentar entrenar SBERT no servia de res. Per això, vam decidir congelar-lo per complet per no trencar els pesos que ja portava preentrenats i centrar tots els recursos en entrenar la part visual i les capes de projecció.

**Paràmetres entrenables de SBERT**: 0 (completament congelat).

**Projecció de text**: `Linear(384→512)` seguida de normalització L2. La projecció és l'únic component del costat de text que s'entrena.

### 2.2 Encoder visual: ResNet50 (parcialment descongelat)

**Model**: `torchvision.models.resnet50` amb pesos preentrenats d'ImageNet.

ResNet50 és una xarxa convolucional residual de 50 capes amb l'arquitectura:

```
conv1 (7×7, stride 2)
bn1 + relu + maxpool
layer1: 3 blocs Bottleneck  (256 ch sortida)
layer2: 4 blocs Bottleneck  (512 ch sortida)
layer3: 6 blocs Bottleneck  (1024 ch sortida)
layer4: 3 blocs Bottleneck  (2048 ch sortida)
avgpool (global average pooling → vector de 2048)
```

La capa de classificació final (`fc`) s'elimina per obtenir el feature map visual pur de **2048 dimensions**.

**Estratègia de descongelació** (`unfreeze_last_layers`): durant l'entrenament (`train.py`), s'inicialitza amb tots els pesos congelats i s'activa el gradient únicament per a **layer4** (índex 7 del `nn.Sequential`), els 3 blocs residuals finals. Això permet al model adaptar les representacions visuals d'alt nivell al domini dels còmics sense modificar les features de baix i mig nivell (textures, contorns, formes) que ja son generals i reutilitzables.

**Paràmetres entrenables de ResNet50**: ~25M total, ~23.5M congelats, ~1.5M entrenables (layer4 + projecció).

**Projecció visual**: `Linear(2048→512)` seguida de normalització L2.

### 2.3 Capes de projecció i espai d'embedding

Tant l'encoder de text com el visual projecten les seves representacions a un espai compartit de **512 dimensions** amb una capa lineal seguida de normalització L2:

```
text_proj : Linear(384  → 512)  +  L2 normalize
image_proj: Linear(2048 → 512)  +  L2 normalize
```

La normalització L2 garanteix que tots els vectors estiguin a la hiperesfera unitària, de forma que la similitud cosinus és equivalent al producte escalar i oscil·la entre -1 i 1. Això simplifica tant l'entrenament com la inferència.

### 2.4 Funció de pèrdua: TripletMarginLoss (pytorch-metric-learning)

**Implementació**: `pytorch_metric_learning.losses.TripletMarginLoss` amb `margin=0.4`.

La Triplet Loss treballa amb tríades `(àncora a, positiu p, negatiu n)` i imposa:

$$\mathcal{L} = \max\bigl(0,\; d(a, p) - d(a, n) + m\bigr)$$

on $d(\cdot, \cdot)$ és la distància euclídea (equivalent al cosinus amb vectors normalitzats) i $m=0.4$ és el marge mínim de separació.

**Construcció dels triplets**: en cada pas d'entrenament es concatenen els embeddings de text i imatge en un sol tensor de `[2N, 512]` i s'assignen labels `[0, 1, ..., N-1, 0, 1, ..., N-1]`, on la parella `(text_i, img_i)` comparteix la mateixa label `i`. El miner automàtic de `pytorch-metric-learning` genera tots els triplets vàlids del batch:

- **Àncora**: `text_i` (o `img_i`)
- **Positiu**: `img_i` (o `text_i`) — mateixa label
- **Negatiu**: qualsevol `img_j` o `text_j` amb `j ≠ i`

Amb batch size `N=64`, hi ha fins a `N×(N-1) = 4032` triplets possibles per pas, dels quals el miner selecciona els **semi-hard negatives** (negatius que ja estan separats però per menys del marge). Això evita tant els negatius trivials (no aporten gradient) com els excessivament difícils (gradient inestable).

**Marge = 0.4**: valor més gran que el típic 0.2, adequat per a un dataset visualment homogeni com COMIC-PAP on panells del mateix còmic son molt similars entre si i cal forçar una separació gran entre parelles incorrectes.

### 2.5 Optimitzador: AdamW

**Configuració**:
- `lr = 1e-4`
- `weight_decay = 0.1`

AdamW és la variant de Adam amb *weight decay* desacoblat de la taxa d'aprenentatge. El weight decay de **0.1** aplica regularització L2 als pesos del model, penalitzant pesos grans i reduint l'overfitting. Vam provar weight_decay=0.01 i 0.1; el valor 0.1 va donar clarament millors resultats al sweep final.

---

## 3. Filtratge del dataset: CLIP Score

Prèviament a l'entrenament, el dataset s'ha filtrat eliminant les parelles text-imatge amb baixa coherència semàntica. Per a cada parella del training set es calcula la **similitud cosinus CLIP** entre el text (primer paràgraf de la caption) i la imatge del panell. Les parelles amb score per sota de **0.29** s'eliminen.

El threshold de 0.29 es va determinar experimentalment: és el valor que maximitza R@10 en el sweep (analitzat sobre la distribució de scores). Elimina les captions que descriuen el context narratiu del panell anterior (no la imatge actual) i conserva les parelles amb alineació semàntica real.

**Important**: el filtratge CLIP s'aplica **exclusivament al training set**. El validation set s'avalua sempre sobre totes les mostres sense filtrar, per garantir que les mètriques siguin representives del rendiment real.

---

## 4. Pipeline d'entrenament

```
Per a cada epoch:
  Per a cada batch de N parells (text_i, img_i):
    1. SBERT(text_i)         → txt_feat [N, 384]    (SBERT congelat)
    2. ResNet50(img_i)        → img_feat [N, 2048]   (layer4 entrenable)
    3. text_proj(txt_feat)    → txt_emb  [N, 512]
    4. image_proj(img_feat)   → img_emb  [N, 512]
    5. L2_normalize(txt_emb)  → txt_norm [N, 512]
    6. L2_normalize(img_emb)  → img_norm [N, 512]
    7. Concatenar: emb = [txt_norm; img_norm]  [2N, 512]
    8. labels = [0..N-1, 0..N-1]
    9. TripletMarginLoss(emb, labels) → loss escalar
   10. optimizer.zero_grad() + loss.backward() + optimizer.step()

  Al final de cada epoch:
    validate_retrieval() → R@1, R@10, MRR sobre tot el val set
    Guardar checkpoint si R@1 millora
    Log a WandB: val/R@1, val/R@10, val/MRR (actual) + val/best_* (millor fins ara)
```

---

## 5. Mètriques d'avaluació

### R@K (Recall at K)

Per a cada consulta textual del val set, es calcula la similitud cosinus amb tots els embeddings d'imatge del val set i s'ordenen de major a menor. R@K és 1 si la imatge correcta apareix entre les K primeres posicions, 0 en cas contrari. La mètrica final és la mitjana sobre totes les consultes.

$$R@K = \frac{1}{|Q|}\sum_{q \in Q} \mathbf{1}[\text{rang}(q) \leq K]$$

### MRR (Mean Reciprocal Rank)

$$MRR = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rang}(q)}$$

MRR és sensible a la posició exacta: un model que sempre retorna la imatge correcta en posició 2 tindrà MRR ≈ 0.5 però R@1 = 0. Complementa R@1 i R@10 donant informació sobre si el model "quasi encerta".

---

## 6. Resum de paràmetres del model final

| Component | Detall |
|---|---|
| Encoder de text | SentenceBERT `all-MiniLM-L6-v2` (384 dim, congelat) |
| Encoder visual | ResNet50 (2048 dim, layer4 entrenable) |
| Espai d'embedding | 512 dimensions, normalitzat L2 |
| Funció de pèrdua | TripletMarginLoss, margin=0.4 |
| Optimitzador | AdamW, lr=1e-4, weight_decay=0.1 |
| Batch size | 64 |
| CLIP threshold | 0.29 (només training) |
| Epochs (run final) | 150 |
| Millor R@1 (sweep) | 0.0458 |
| Millor R@10 (sweep) | 0.2328 |
| Millor MRR (sweep) | 0.1002 |

---

## 7. Resultats obtinguts i validació del disseny

### 7.1 Mètriques finals (run de 150 epochs)

| Mètrica | Valor | Baseline (Atzar Real) | Rendiment sobre l'Atzar |
|---|---|---|---|
| R@1 | **0.0458** | ~0.0038 ($1/262$) | **×12.0** |
| R@10 | **0.1946** | ~0.0381 ($10/262$) | **×5.1** |
| MRR | **0.0903** | ~0.0212 | **×4.3** |

### 7.2 Per què l'arquitectura produeix aquests resultats

**SentenceBERT congelat** serveix per treure els vectors de text fixos des del principi. Com vam veure en l'anàlisi del dataset que els textos tenien falles d'origen i no coincidien bé amb les imatges, congelar SBERT evita que el model es torni boig intentant aprendre text sorollós. Així, deixem que la capa lineal `Linear(384→512)` s'encarregui simplement de passar aquests vectors a l'espai compartit.

**ResNet50 amb layer4 entrenable** adapta de manera òptima les representacions de contingut conceptual, composició i disposició de vinyetes al domini dels còmics sense destruir les features generals de baix nivell (línies, contorns i colors) retingudes a les capes congelades inferiors. Amb 2048 features estructurals de sortida, la projecció visual té la informació necessària per condensar-se en les 512 dimensions de destí.

**TripletMarginLoss amb marge 0.4** estructura l'espai mètric forçant que els parells text-imatge correctes s'apropin i els incorrectes se separin mitjançant una penalització geomètrica exigent. L'ús de `margin=0.4` és clau per combatre l'homogeneïtat de l'estil visual dels còmics.

**Normalització L2** a la sortida de les dues branques de projecció és l'eix central del bi-encoder: restringeix els vectors a la superfície d'una hiperesfera unitària, assegurant que el producte escalar esdevingui la similitud cosinus pura i que les distàncies geomètriques a l'espai compartit siguin completament comparables durant el retrieval.

**CLIP filtering a 0.29** actua com un excel·lent filtre de soroll al training set, eliminant aquelles captions que afegeixen desalineació semàntica (com descripcions contextuals de vinyetes passades), garantint que el model aprengui exclusivament amb parelles d'alta correspondència conceptual.

### 7.3 Anàlisi de la convergència

La train/epoch_loss baixa de 0.35 (epoch 1) a 0.087 (epoch 150) de forma consistent, sense oscillació ni divergència. Això confirma que:
- El learning rate de 1e-4 és adequat (no massa gran → inestabilitat, no massa petit → convergència massa lenta)
- El weight decay de 0.1 regularitza correctament sense impedir l'aprenentatge
- L'estratègia de descongelació (layer4 des del primer epoch) és estable
