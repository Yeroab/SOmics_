# somics_docs.py — Documentation strings for SOmics-ML
# Edit this file to update documentation without touching the main app.

OVERVIEW = """
### Purpose and Scope

SOmics-ML is a spatial transcriptomics analysis tool designed to quantify the
CAF-Immune axis in ovarian cancer tissue. The system applies a validated machine
learning model to 10x Visium spatial gene expression data, assigning each tissue
spot a continuous score representing its position on the spectrum from
Cancer-Associated Fibroblast (CAF) dominance to Immune cell dominance.

The tool is intended to support exploratory spatial analysis and hypothesis
generation. Scores reflect patterns learned from the training cohort and should
be interpreted alongside histological and clinical context.

---

### Biological Context

Ovarian cancer is characterised by a highly immunosuppressive tumour
microenvironment. Two dominant niche types compete across the tissue:

**CAF-high niches (score near 0)** are stromal-dense regions where
cancer-associated fibroblasts deposit extracellular matrix, suppress immune
infiltration, and promote tumour invasion. These regions are associated with
resistance to immunotherapy and poor prognosis.

**Immune-high niches (score near 1)** are regions of active immune infiltration,
characterised by cytotoxic T cell and NK cell presence. Spatial proximity of
immune-high niches to tumour regions is associated with better treatment response.

The 1,000-gene signature used by SOmics-ML was derived from single-cell and
spatial transcriptomics analysis of high-grade serous ovarian cancer (HGSOC)
specimens and captures the transcriptional programs that distinguish these two
niche states.

---

### Analysis Pipeline

**1. Data Ingestion**
Raw 10x Visium output files are read directly in MTX mode (recommended), or a
pre-converted expression CSV is accepted in CSV mode. MTX mode reads the sparse
count matrix, feature IDs, and barcodes directly from Space Ranger output.

**2. In-Tissue Filtering**
Spots are filtered to those marked as in-tissue in the tissue positions file
(in_tissue == 1). Off-tissue spots are excluded from all downstream steps.

**3. Normalisation**
In MTX mode, raw UMI counts are normalised to counts per million (CPM) and
log1p-transformed — the same preprocessing applied during model training. This
step is critical for accurate predictions. CSV mode assumes the data has already
been normalised.

**4. Feature Alignment**
The normalised expression matrix is aligned to the model's 1,000-gene feature
space. Genes present in the data but not in the model are ignored. Genes required
by the model but absent from the data are filled with zero.

**5. Inference**
The selected model (Random Forest or Logistic Regression) outputs a probability
score between 0 and 1 for each spot. This represents P(Immune-high), where
values above 0.5 indicate immune-dominant spots and values below 0.5 indicate
CAF-dominant spots.

**6. Visualisation**
Results are rendered as an interactive spatial scatter plot. If a tissue image is
provided, spots are overlaid directly on the histology. The score distribution
histogram and summary metrics are shown alongside the spatial map.
"""

MODEL_ARCH = """
### Machine Learning Models

SOmics-ML offers two classification models trained on the same 1,000-gene spatial
transcriptomics feature set. Both models output a continuous probability score
rather than a hard binary label, allowing for finer spatial interpretation.

---

### Random Forest

The Random Forest model is the default and recommended choice. It is an ensemble
of decision trees trained using bootstrap aggregation (bagging). Each tree is
trained on a random subset of training spots and a random subset of features at
each split. The final probability score is the average of predictions across all
trees in the ensemble.

Random Forest is preferred for this application because it is robust to outlier
expression values, handles the sparse and noisy nature of spatial transcriptomics
data well, and provides interpretable feature importance scores via the mean
decrease in impurity metric. It is also less sensitive to the scale of input
features than linear models, making it more forgiving when normalisation is
imperfect.

---

### Logistic Regression

The Logistic Regression model is a regularised linear classifier. It learns a
single weight for each of the 1,000 features and outputs a probability via the
logistic (sigmoid) function. It is faster to run than the Random Forest and
produces more interpretable coefficients — positive weights push toward
Immune-high, negative weights push toward CAF-high.

Logistic Regression is recommended when interpretability at the individual gene
level is the primary goal, or when computational speed is a concern for large
datasets. It is more sensitive to input scale, so correct log1p CPM
normalisation is especially important in this mode.

---

### Feature Set

Both models operate on the same 1,000-gene signature. These genes were selected
through differential expression analysis comparing CAF-high and immune-high spots
across a training cohort of high-grade serous ovarian cancer 10x Visium samples.
Genes were ranked by their ability to discriminate between the two niche states
and filtered for spatial consistency across samples.

All features are identified by Ensembl gene ID. The app strips version suffixes
automatically (e.g. ENSG00000001234.5 is treated as ENSG00000001234) to ensure
compatibility across reference genome versions.

---

### Score Interpretation

| Score Range | Classification | Biological Interpretation |
|------------|---------------|--------------------------|
| 0.00 - 0.30 | CAF-high | Strongly stromal-dominant niche |
| 0.30 - 0.50 | CAF-high (borderline) | Stromal tendency, mixed signals |
| 0.50 - 0.70 | Immune-high (borderline) | Immune tendency, mixed signals |
| 0.70 - 1.00 | Immune-high | Strongly immune-infiltrated niche |

Borderline spots (0.30 - 0.70) often correspond to transitional zones at the
tumour-stroma interface and are biologically meaningful — they should not be
dismissed as uncertain classifications but examined spatially in relation to
neighbouring high-confidence spots.

---

### Training Data

Both models were trained on 10x Visium spatial transcriptomics data from
high-grade serous ovarian cancer (HGSOC) specimens. Expression data was
normalised to log1p CPM before training. Spot labels were derived from
deconvolution of cell type signatures estimated from matched single-cell
RNA-seq data.
"""

GUI_GUIDE = """
### Navigating the Application

The application has four pages accessible from the left sidebar:

- **Home** — Overview of the biological context and tool capabilities.
- **Demo Walkthrough** — Runs the full pipeline on a bundled real ovarian cancer
  tissue sample. No file upload required.
- **User Analysis** — The main workspace for analysing your own 10x Visium data.
- **Documentation** — This page.

---

### Demo Walkthrough Page

The demo page runs the complete SOmics-ML pipeline on a real ovarian cancer biopsy
processed through 10x Visium spatial transcriptomics. All data files are bundled
with the app — no upload is required.

Select either the Random Forest or Logistic Regression model and click
"Run Demo Analysis". The pipeline applies log1p CPM normalisation, aligns to the
1,000-gene feature space, runs inference, and displays the scored spots overlaid
on the tissue lowres image. Results persist while you interact with the page.

A score distribution histogram and four summary metrics are shown below the
spatial map. Click "Download Demo Results" to export the barcode-level scores as
a CSV. Click "Reset Demo" to clear results and run again with a different model.

---

### User Analysis Page

The User Analysis page has two input modes selectable at the top of the page.

**MTX Mode (recommended)**

MTX mode reads directly from 10x Visium Space Ranger output files. It applies
the same log1p CPM normalisation used during model training, producing the most
accurate predictions. Upload the following files:

- matrix.mtx or matrix.mtx.gz — the sparse UMI count matrix
- features.tsv or features.tsv.gz — gene IDs (Ensembl format)
- barcodes.tsv or barcodes.tsv.gz — spot barcodes
- tissue_positions.csv or tissue_positions_list.csv — spot pixel coordinates

**CSV Mode**

CSV mode accepts a pre-converted expression matrix where spots are rows and
Ensembl gene IDs are columns. No normalisation is applied — the data is assumed
to already be in log1p CPM format. This mode is provided for compatibility with
downstream pipelines that have already processed the raw data.

---

### Tissue Image Overlay

Both modes support an optional tissue image upload for spatial visualisation.
The tissue image is overlaid with coloured spots representing CAF-Immune scores.

Upload either the lowres image (tissue_lowres_image.png, ~600px) or the
full-resolution image from your Space Ranger spatial folder.

If you upload the lowres image, also upload scalefactors_json.json — the scale
factor (tissue_lowres_scalef) is read automatically and applied to the pixel
coordinates. If you upload the full-resolution image, set the image resolution
toggle to "Full-resolution" and no scaling is applied.

If no image is uploaded, a plain spatial scatter plot is shown using pixel
coordinates from the positions file.

Spot size and opacity can be adjusted using the sliders that appear when an image
is uploaded.

---

### Input File Format Reference

**tissue_positions_list.csv (Space Ranger < 2.0 — no header)**
```
AAACAAGTATCTCCCA-1,1,50,102,7474,8500
AAACATTTCCCGGATT-1,1,51,103,7624,8644
```
Columns: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col

**tissue_positions.csv (Space Ranger >= 2.0 — has header)**
```
barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
AAACAAGTATCTCCCA-1,1,50,102,7474,8500
```
Both formats are detected and parsed automatically.

**Expression CSV (CSV mode)**
```
,ENSG00000243485,ENSG00000237613
AAACAAGTATCTCCCA-1,0.00,1.24
AAACATTTCCCGGATT-1,2.31,0.00
```
First column is the barcode index. Gene columns should be Ensembl IDs.
Prefixed format (RNA_ENSG...) is also accepted.

---

### Common Errors

**"Demo data files not found"** — The demo_data/ folder is missing from the repo
root. See DEMO_DATA_README.md for the required folder structure and file names.

**"Error during processing"** in MTX mode — Most commonly caused by uploading
files from different samples (mismatched barcodes between matrix and positions),
or uploading the raw_feature_bc_matrix folder instead of filtered_feature_bc_matrix.
Always use the filtered matrix for spatial analysis.

**Spots not aligned to tissue in overlay** — The wrong scale factor is being
applied. If spots appear clustered in one corner, check that you selected the
correct image resolution (full-res vs downsampled) and that the scale factor
matches the image you uploaded. Upload scalefactors_json.json to have this
resolved automatically.

**All spots scoring near 0.5** — May indicate that few of the 1,000 model genes
were found in your data. Check that your features file uses Ensembl IDs and that
your data is from a human sample aligned to a human reference genome.
"""
