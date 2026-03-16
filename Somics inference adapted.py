"""
Adapted from somics_spatial_inference.py for Streamlit GUI use.
Works with uploaded file bytes instead of filesystem paths.
"""

import io
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def log1p_cpm(counts):
    """Normalize sparse count matrix to log1p CPM."""
    c = counts.tocsc(copy=True)
    col_sums = np.array(c.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    c = c.multiply(1e6 / col_sums)
    c = c.tocsr()
    c.data = np.log1p(c.data)
    return c


def run_inference_from_bytes(mtx_bytes, features_bytes, barcodes_bytes, pos_df, model, model_features):
    """
    Run SOmics inference on uploaded file bytes.

    Args:
        mtx_bytes: Raw bytes from matrix.mtx(.gz) file
        features_bytes: Raw bytes from features.tsv(.gz) file
        barcodes_bytes: Raw bytes from barcodes.tsv(.gz) file
        pos_df: Pandas DataFrame with tissue positions
        model: Trained classifier model
        model_features: List of gene IDs (in order) that model expects

    Returns:
        DataFrame with barcodes, coordinates, and prediction scores
    """
    # Parse sparse matrix
    counts = mmread(io.BytesIO(mtx_bytes)).tocsr()

    # Parse feature IDs — strip Ensembl version decimals
    feat_lines = features_bytes.decode('utf-8').strip().split('\n')
    feature_ids = [line.split('\t')[0].split('.')[0] for line in feat_lines if line]

    # Parse barcodes
    bc_lines = barcodes_bytes.decode('utf-8').strip().split('\n')
    barcode_ids = [line.strip() for line in bc_lines if line]

    # Filter to in-tissue spots
    pos_df = pos_df[pos_df['in_tissue'] == 1].copy()
    barcode_to_index = {b: i for i, b in enumerate(barcode_ids)}
    keep_indices = [barcode_to_index[b] for b in pos_df['barcode'] if b in barcode_to_index]
    barcodes_kept = [barcode_ids[i] for i in keep_indices]

    # Select in-tissue spot columns (counts is genes × spots)
    counts = counts[:, keep_indices]

    # Normalize
    counts = log1p_cpm(counts)

    # Build gene index map
    vis_genes = [g.split('.')[0] for g in feature_ids]
    model_genes = [g.split('.')[0] for g in model_features]

    gene_map = {}
    for i, g in enumerate(vis_genes):
        if g not in gene_map:
            gene_map[g] = i

    # Align to model features (sparse column assembly)
    X_cols = []
    for g in model_genes:
        if g in gene_map:
            X_cols.append(counts[gene_map[g], :].T)
        else:
            X_cols.append(sparse.csr_matrix((counts.shape[1], 1)))

    X = sparse.hstack(X_cols).tocsr()

    # Run prediction
    probs = model.predict_proba(X)[:, 1]

    # Attach scores to position dataframe
    pos_df = pos_df[pos_df['barcode'].isin(barcodes_kept)].copy()
    score_series = pd.Series(probs, index=barcodes_kept)
    pos_df['Score'] = score_series.reindex(pos_df['barcode']).values

    return pos_df