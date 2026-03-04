import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import gzip
import base64
import io
from scipy import sparse
from scipy.io import mmread
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tifffile
from somics_docs import OVERVIEW, MODEL_ARCH, GUI_GUIDE

# ==========================================
# 1. PAGE SETUP & THEME
# ==========================================
st.set_page_config(page_title="SOmics-ML: CAF-Immune", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #40E0D0; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #20B2AA; text-align: center; margin-bottom: 2rem; }
    .stMetric { background-color: #E0F7FA; padding: 15px; border-radius: 10px; border-left: 5px solid #40E0D0; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #E0F7FA 0%, #B2EBF2 100%); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_assets():
    try:
        rf_model = joblib.load('somics_rf (1).pkl')['model']
        lr_model = joblib.load('somics_lr (1).pkl')['model']
        with open('model_features_1000 (1).json', 'r') as f:
            features_data = json.load(f)
            model_features = features_data['model_features_ordered']
        with open('hub_genes.json', 'r') as f:
            hub_genes_data = json.load(f)
        return rf_model, lr_model, model_features, hub_genes_data
    except Exception as e:
        st.error(f"Error loading model files: {e}. Ensure all required files are in the root directory.")
        return None, None, None, None

rf_model, lr_model, model_features, hub_genes_data = load_assets()
assets_loaded = all(x is not None for x in [rf_model, lr_model, model_features, hub_genes_data])

# ==========================================
# 3. INFERENCE PIPELINE
# Adapted from somics_spatial_inference.py
# Two modes:
#   A. MTX mode  — reads raw 10x Visium folder structure, applies log1p CPM normalisation
#   B. CSV mode  — legacy path for pre-converted expression CSV (no normalisation)
# ==========================================

def log1p_cpm(counts):
    """Normalise a sparse count matrix to log1p CPM (as used during model training)."""
    c = counts.tocsc(copy=True)
    col_sums = np.array(c.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    c = c.multiply(1e6 / col_sums)
    c = c.tocsr()
    c.data = np.log1p(c.data)
    return c


def run_inference_mtx(mtx_bytes, features_bytes, barcodes_bytes, pos_df, model, model_feats):
    """
    Full pipeline from raw MTX files — mirrors somics_spatial_inference.py.
    Accepts file bytes (from Streamlit uploaders) rather than filesystem paths.

    Steps:
      1. Parse MTX, features, barcodes from bytes
      2. Filter to in-tissue spots using pos_df
      3. Apply log1p CPM normalisation
      4. Align to model feature space (sparse, zero-fill missing genes)
      5. Predict probabilities
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

    # counts is (genes x spots) in MTX format — select in-tissue spot columns
    counts = counts[:, keep_indices]

    # Normalise
    counts = log1p_cpm(counts)

    # Build gene index map (first occurrence wins for duplicate IDs)
    gene_map = {}
    for i, g in enumerate(feature_ids):
        if g not in gene_map:
            gene_map[g] = i

    # Align to model features (sparse column assembly)
    model_genes = [g.split('.')[0] for g in model_feats]
    X_cols = []
    for g in model_genes:
        if g in gene_map:
            X_cols.append(counts[gene_map[g], :].T)
        else:
            X_cols.append(sparse.csr_matrix((counts.shape[1], 1)))
    X = sparse.hstack(X_cols).tocsr()

    probs = model.predict_proba(X)[:, 1]

    # Attach scores back to position dataframe
    score_series = pd.Series(probs, index=barcodes_kept)
    pos_df = pos_df[pos_df['barcode'].isin(barcodes_kept)].copy()
    pos_df['Score'] = score_series.reindex(pos_df['barcode']).values
    return pos_df


def run_inference_csv(df, model, features):
    """
    Legacy CSV path. Works on a copy to avoid mutating the caller's dataframe.
    Strips Ensembl version suffixes from column names selectively.
    No log1p CPM normalisation — assumes the CSV was pre-normalised or is being
    used for quick testing only.
    """
    df = df.copy()
    df.columns = [
        str(c).split('.')[0] if str(c).startswith('ENSG') else str(c)
        for c in df.columns
    ]
    X = df.reindex(columns=features, fill_value=0.0)
    probs = model.predict_proba(X)[:, 1]
    return probs


# ==========================================
# 4. IMAGE HELPERS
# ==========================================

def load_tissue_image(uploaded_file):
    """
    Return a PIL Image from either a JPEG/PNG or a TIFF.
    tifffile handles high-bit-depth and multi-page TIFFs that PIL cannot open.
    """
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        arr = tifffile.imread(io.BytesIO(raw))
        # Some TIFF writers store axes as (C, H, W) — reorder to (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[1]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        return Image.open(io.BytesIO(raw))


def overlay_spots_on_image(pil_image, final_df, scale_factor=1.0, spot_opacity=0.85, spot_size=8):
    """
    Build a Plotly figure with the tissue histology image as background and
    CAF-Immune scored spots overlaid at their pixel coordinates.

    Coordinate system:
    - Plotly layout images with yanchor='bottom' sit at y=0 and extend UP to y=img_h.
    - The axis range is [0, img_h] with y=0 at the bottom.
    - Pixel coordinates from 10x Visium use image convention: row 0 is at the TOP.
    - So we flip y: y_plot = img_h - (pxl_row * scale_factor)
    - x is unchanged: x_plot = pxl_col * scale_factor
    """
    img_w, img_h = pil_image.size

    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    x_coords = final_df['pxl_col'] * scale_factor
    # Flip y so image-space row 0 (top) maps to Plotly y=img_h (top of axis)
    y_coords = img_h - (final_df['pxl_row'] * scale_factor)

    fig = go.Figure()

    # Place image at y=0 (bottom of axis), extending upward by img_h
    fig.add_layout_image(
        source=b64,
        x=0, y=0,
        xref="x", yref="y",
        sizex=img_w, sizey=img_h,
        xanchor="left", yanchor="bottom",
        layer="below",
        opacity=1.0
    )

    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            color=final_df['Score'],
            colorscale=[[0, "#FF6B6B"], [0.5, "#FFFFFF"], [1, "#40E0D0"]],
            cmin=0, cmax=1,
            size=spot_size,
            opacity=spot_opacity,
            colorbar=dict(title="Immune Score"),
            line=dict(width=0),
        ),
        text=final_df['barcode'],
        hovertemplate="<b>%{text}</b><br>Score: %{marker.color:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(range=[0, img_w], showgrid=False, zeroline=False, visible=False),
        # Normal y axis (0 at bottom, img_h at top) — image and spots use same space
        yaxis=dict(range=[0, img_h], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title="CAF-Immune Spatial Map — Tissue Overlay",
        plot_bgcolor="black",
    )
    return fig


def parse_positions(pos_file_bytes, filename):
    """
    Parse tissue positions file. Handles both:
      - tissue_positions_list.csv (no header, 6 columns — Space Ranger < 2.0)
      - tissue_positions.csv      (has header — Space Ranger >= 2.0)
    Also normalises column names for pxl_row_in_fullres -> pxl_row convention.
    """
    try:
        pos = pd.read_csv(io.BytesIO(pos_file_bytes))
        # If first column is unnamed or the file has no recognisable header,
        # treat it as the headerless format
        if pos.columns[0].startswith('Unnamed') or pos.columns[0] not in [
                'barcode', 'Barcode', 'barcodes']:
            raise ValueError("No header detected")
    except Exception:
        pos = pd.read_csv(io.BytesIO(pos_file_bytes), header=None)
        pos.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']

    # Normalise barcode column name
    bc_candidates = ['barcode', 'Barcode', 'barcodes', 'spot_id']
    bc_col = next((c for c in bc_candidates if c in pos.columns), pos.columns[0])
    if bc_col != 'barcode':
        pos = pos.rename(columns={bc_col: 'barcode'})

    # Normalise pixel coordinate column names (Space Ranger >= 2.0 convention)
    if 'pxl_row_in_fullres' in pos.columns:
        pos = pos.rename(columns={
            'pxl_row_in_fullres': 'pxl_row',
            'pxl_col_in_fullres': 'pxl_col'
        })

    # Add in_tissue=1 for all rows if the column is missing (some exports omit it)
    if 'in_tissue' not in pos.columns:
        pos['in_tissue'] = 1

    return pos


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## SOmics-ML")
    page = st.radio("Go to:", ["Home", "Demo Walkthrough", "Classify - User Analysis", "Documentation"])

# ==========================================
# 6. PAGE: HOME
# ==========================================
if page == "Home":
    st.markdown('<div class="main-header">SOmics-ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Spatial Analysis of the CAF-Immune Axis</div>',
                unsafe_allow_html=True)

    # Display method diagram
    try:
        method_image = Image.open('method.png')
        st.image(method_image, use_container_width=True)
    except FileNotFoundError:
        st.error("method.png not found. Please ensure the file is in the same directory as this script.")
    except Exception as e:
        st.error(f"Error loading method.png: {e}")

# ==========================================
# 7. PAGE: DEMO WALKTHROUGH
# ==========================================
elif page == "Demo Walkthrough":
    st.markdown('<div class="main-header">Interactive Demo</div>', unsafe_allow_html=True)
    st.write("""
    This demo runs the full SOmics-ML pipeline on a real ovarian cancer spatial
    transcriptomics sample. All files are bundled with the app — no upload required.
    """)

    if not assets_loaded:
        st.error("Model assets could not be loaded. Cannot run demo.")
        st.stop()

    # ------------------------------------------------------------------
    # Demo file loader — reads the bundled spatial data files that ship
    # with the repo (same files as the real sample used for validation).
    # Expected repo layout:
    #   demo_data/
    #     matrix.mtx.gz
    #     features.tsv.gz
    #     barcodes.tsv.gz
    #     tissue_positions_list.csv
    #     tissue_lowres_image.png
    #     scalefactors_json.json
    # ------------------------------------------------------------------
    DEMO_DIR = "demo_data"

    @st.cache_data
    def load_demo_results(_rf_model, _lr_model, _model_features, model_type="Random Forest"):
        """
        Run the full MTX pipeline on the bundled demo data and return the
        results dataframe and the lowres image. Cached so it only runs once
        per model selection.
        """
        import os

        def read_gz(path):
            with gzip.open(path, 'rb') as f:
                return f.read()

        raw_mtx  = read_gz(os.path.join(DEMO_DIR, "matrix.mtx.gz"))
        raw_feat = read_gz(os.path.join(DEMO_DIR, "features.tsv.gz"))
        raw_bc   = read_gz(os.path.join(DEMO_DIR, "barcodes.tsv.gz"))

        pos_path1 = os.path.join(DEMO_DIR, "tissue_positions_list.csv")
        pos_path2 = os.path.join(DEMO_DIR, "tissue_positions.csv")
        pos_bytes = open(pos_path1 if os.path.exists(pos_path1) else pos_path2, 'rb').read()
        pos_df    = parse_positions(pos_bytes, "tissue_positions_list.csv")

        with open(os.path.join(DEMO_DIR, "scalefactors_json.json")) as f:
            sf = json.load(f)
        scale_factor = sf.get("tissue_lowres_scalef", 0.05)

        model = _rf_model if model_type == "Random Forest" else _lr_model
        final_df = run_inference_mtx(raw_mtx, raw_feat, raw_bc, pos_df, model, _model_features)

        img = Image.open(os.path.join(DEMO_DIR, "tissue_lowres_image.png"))

        return final_df, img, scale_factor

    # Model selector
    demo_model = st.radio(
        "Model", ["Random Forest", "Logistic Regression"], horizontal=True
    )

    if st.button("Run Demo Analysis", type="primary"):
        with st.spinner("Running pipeline on real ovarian cancer tissue sample..."):
            try:
                final_df, demo_img, scale_factor = load_demo_results(
                    rf_model, lr_model, model_features, demo_model
                )
                st.session_state.demo_results      = final_df
                st.session_state.demo_img          = demo_img
                st.session_state.demo_scale        = scale_factor
                st.session_state.demo_model_used   = demo_model
            except FileNotFoundError as e:
                st.error(
                    f"Demo data files not found: {e}\n\n"
                    f"Ensure the `{DEMO_DIR}/` folder is present in your repo root "
                    f"containing matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz, "
                    f"tissue_positions_list.csv, tissue_lowres_image.png, and "
                    f"scalefactors_json.json."
                )

    if 'demo_results' in st.session_state:
        final_df     = st.session_state.demo_results
        demo_img     = st.session_state.demo_img
        scale_factor = st.session_state.demo_scale

        # --- tissue overlay plot ---
        fig = overlay_spots_on_image(
            demo_img, final_df,
            scale_factor=scale_factor,
            spot_opacity=0.80,
            spot_size=6
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Real ovarian cancer tissue — {len(final_df)} in-tissue spots  |  "
            f"Model: {st.session_state.demo_model_used}  |  "
            f"Scale factor: {scale_factor:.5f} (tissue_lowres_scalef)"
        )

        st.divider()
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Total Spots", len(final_df))
        with col_d2:
            immune_n = (final_df['Score'] > 0.5).sum()
            st.metric("Immune-high Spots",
                      f"{immune_n} ({immune_n/len(final_df):.1%})")
        with col_d3:
            caf_n = (final_df['Score'] <= 0.5).sum()
            st.metric("CAF-high Spots",
                      f"{caf_n} ({caf_n/len(final_df):.1%})")
        with col_d4:
            st.metric("Mean Score", f"{final_df['Score'].mean():.3f}")

        # Score distribution
        st.divider()
        col_hist, col_info = st.columns([2, 1])
        with col_hist:
            fig_hist = px.histogram(
                final_df, x='Score', nbins=40,
                color_discrete_sequence=["#40E0D0"],
                title="Distribution of CAF-Immune Scores Across Spots",
                labels={'Score': 'Immune Score (0=CAF-high, 1=Immune-high)'}
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="gray",
                               annotation_text="Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_info:
            st.markdown("### About this sample")
            st.write("""
            This is a real ovarian cancer biopsy processed through 10x Visium
            spatial transcriptomics.

            Spots are scored on a continuous axis:
            - **Score near 0** — CAF-dominant (coral)
            - **Score near 1** — Immune-dominant (turquoise)

            The spatial distribution reflects the immunosuppressive niche
            architecture characteristic of high-grade ovarian cancer.
            """)

        with st.expander("Download Demo Results"):
            csv_out = final_df[['barcode', 'Score', 'pxl_row', 'pxl_col']].to_csv(
                index=False).encode('utf-8')
            st.download_button(
                "Download scores CSV", csv_out,
                file_name="somics_demo_scores.csv", mime="text/csv"
            )

        if st.button("Reset Demo"):
            for k in ['demo_results', 'demo_img', 'demo_scale', 'demo_model_used']:
                st.session_state.pop(k, None)
            st.rerun()

# ==========================================
# 8. PAGE: CLASSIFY - USER ANALYSIS
# ==========================================
elif page == "Classify - User Analysis":
    st.markdown('<div class="main-header">Classify - User Analysis</div>', unsafe_allow_html=True)

    if not assets_loaded:
        st.error("Model assets could not be loaded. Cannot run analysis.")
        st.stop()

    # ---------- input mode toggle ----------
    input_mode = st.radio(
        "Input mode",
        ["MTX (raw 10x Visium)", "CSV (pre-converted)"],
        horizontal=True,
        help=(
            "MTX mode reads directly from 10x Visium output files and applies "
            "the same log1p CPM normalisation used during model training. "
            "CSV mode accepts a pre-converted expression matrix — no normalisation is applied."
        )
    )

    col_u1, col_u2 = st.columns([1, 2])

    with col_u1:
        model_type = st.selectbox("Model", ["Random Forest", "Logistic Regression"])

        if input_mode == "MTX (raw 10x Visium)":
            st.markdown("**Upload 10x Visium files**")
            mtx_file      = st.file_uploader("matrix.mtx or matrix.mtx.gz",   type=['mtx', 'gz'])
            feat_file     = st.file_uploader("features.tsv or features.tsv.gz", type=['tsv', 'gz'])
            bc_file       = st.file_uploader("barcodes.tsv or barcodes.tsv.gz", type=['tsv', 'gz'])
            pos_file      = st.file_uploader("tissue_positions.csv (or _list.csv)", type=['csv'])
            sf_file       = st.file_uploader(
                "scalefactors_json.json (optional — auto-reads scale for lowres image)",
                type=['json']
            )
            image_file    = st.file_uploader(
                "Tissue image — optional (.jpg, .png, .tif/.tiff)",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff']
            )
            expr_file = None  # not used in MTX mode

        else:  # CSV mode
            st.markdown("**Upload CSV files**")
            expr_file  = st.file_uploader("Expression CSV (spots x genes, Ensembl IDs)", type=['csv'])
            pos_file   = st.file_uploader("Tissue positions CSV", type=['csv'])
            sf_file    = st.file_uploader(
                "scalefactors_json.json (optional — auto-reads scale for lowres image)",
                type=['json']
            )
            image_file = st.file_uploader(
                "Tissue image — optional (.jpg, .png, .tif/.tiff)",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff']
            )
            mtx_file = feat_file = bc_file = None  # not used in CSV mode

        # Scale factor controls — only shown if image uploaded and no scalefactors JSON
        if image_file is not None:
            if sf_file is not None:
                st.info("Scale factor will be read automatically from scalefactors_json.json.")
                scale_factor = None  # resolved at run time from JSON
            else:
                image_res = st.radio(
                    "Image resolution",
                    ["Full-resolution", "Downsampled (lowres/hires)"],
                    help=(
                        "Full-resolution: pixel coordinates map directly onto this image.\n\n"
                        "Downsampled: provide the scale factor from scalefactors_json.json, "
                        "or upload that file above to have it read automatically."
                    )
                )
                if image_res == "Downsampled (lowres/hires)":
                    scale_factor = st.number_input(
                        "Scale factor",
                        min_value=0.001, max_value=1.0,
                        value=0.05, step=0.001, format="%.3f",
                        help="tissue_lowres_scalef ≈ 0.05, tissue_hires_scalef ≈ 0.20"
                    )
                else:
                    scale_factor = 1.0
            spot_size    = st.slider("Spot size",    3, 20,  8)
            spot_opacity = st.slider("Spot opacity", 0.1, 1.0, 0.85, step=0.05)
        else:
            scale_factor = 1.0
            spot_size    = 8
            spot_opacity = 0.85

    # ---------- determine if ready to run ----------
    mtx_ready = input_mode == "MTX (raw 10x Visium)" and all(
        f is not None for f in [mtx_file, feat_file, bc_file, pos_file]
    )
    csv_ready = input_mode == "CSV (pre-converted)" and all(
        f is not None for f in [expr_file, pos_file]
    )

    if mtx_ready or csv_ready:
        try:
            active_model = rf_model if model_type == "Random Forest" else lr_model

            if st.button("Run Prediction", type="primary"):
                with st.spinner("Running inference..."):

                    # --- parse positions (shared by both modes) ---
                    pos_bytes = pos_file.read()
                    pos_df = parse_positions(pos_bytes, pos_file.name)

                    if input_mode == "MTX (raw 10x Visium)":
                        # Read MTX — handle .gz transparently
                        raw_mtx  = mtx_file.read()
                        raw_feat = feat_file.read()
                        raw_bc   = bc_file.read()

                        if mtx_file.name.endswith('.gz'):
                            raw_mtx = gzip.decompress(raw_mtx)
                        if feat_file.name.endswith('.gz'):
                            raw_feat = gzip.decompress(raw_feat)
                        if bc_file.name.endswith('.gz'):
                            raw_bc = gzip.decompress(raw_bc)

                        final_df = run_inference_mtx(
                            raw_mtx, raw_feat, raw_bc,
                            pos_df, active_model, model_features
                        )
                        # Rename pCAF -> Score for consistent downstream use
                        if 'pCAF' in final_df.columns and 'Score' not in final_df.columns:
                            final_df = final_df.rename(columns={'pCAF': 'Score'})

                    else:  # CSV mode
                        expr_file.seek(0)
                        df_expr  = pd.read_csv(expr_file, index_col=0)
                        scores   = run_inference_csv(df_expr, active_model, model_features)
                        results  = pd.DataFrame({'barcode': df_expr.index, 'Score': scores})
                        final_df = pd.merge(results, pos_df, on='barcode')

                    # --- resolve scale factor from JSON if provided ---
                    resolved_scale = scale_factor
                    if sf_file is not None:
                        sf_file.seek(0)
                        sf_data = json.load(sf_file)
                        # Use lowres scale if image name suggests lowres, else hires
                        if image_file is not None and 'hires' in image_file.name.lower():
                            resolved_scale = sf_data.get('tissue_hires_scalef', 1.0)
                        else:
                            resolved_scale = sf_data.get('tissue_lowres_scalef', 0.05)

                    st.session_state.live_results      = final_df
                    st.session_state.live_model_type   = model_type
                    st.session_state.live_scale_factor = resolved_scale
                    st.session_state.live_spot_size    = spot_size
                    st.session_state.live_spot_opacity = spot_opacity

                    if image_file is not None:
                        image_file.seek(0)
                        st.session_state.live_image_bytes = image_file.read()
                        st.session_state.live_image_name  = image_file.name
                    else:
                        for k in ['live_image_bytes', 'live_image_name']:
                            st.session_state.pop(k, None)

            # ---------- render results ----------
            if 'live_results' in st.session_state:
                final_df = st.session_state.live_results

                with col_u2:
                    if 'live_image_bytes' in st.session_state:
                        img_buf = io.BytesIO(st.session_state.live_image_bytes)
                        img_buf.name = st.session_state.live_image_name
                        pil_img = load_tissue_image(img_buf)
                        fig = overlay_spots_on_image(
                            pil_img, final_df,
                            scale_factor=st.session_state.live_scale_factor,
                            spot_opacity=st.session_state.live_spot_opacity,
                            spot_size=st.session_state.live_spot_size,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        img_w, img_h = pil_img.size
                        st.caption(
                            f"Image: {img_w} x {img_h} px  |  "
                            f"Scale factor: {st.session_state.live_scale_factor}  |  "
                            f"{len(final_df)} spots overlaid"
                        )
                    else:
                        fig = px.scatter(
                            final_df, x='pxl_col', y='pxl_row', color='Score',
                            color_continuous_scale=["#FF6B6B", "#FFFFFF", "#40E0D0"],
                            title=f"CAF-Immune Spatial Map ({st.session_state.live_model_type})",
                            labels={'Score': 'Immune Score', 'pxl_col': 'X', 'pxl_row': 'Y'}
                        )
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)

                st.divider()
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Total Spots", len(final_df))
                with col_r2:
                    immune_n = (final_df['Score'] > 0.5).sum()
                    st.metric("Immune-high Spots",
                              f"{immune_n} ({immune_n/len(final_df):.1%})")
                with col_r3:
                    st.metric("Mean Score", f"{final_df['Score'].mean():.3f}")

                with st.expander("Download Results"):
                    out_cols = ['barcode', 'Score']
                    if 'CAF_high' in final_df.columns:
                        out_cols.append('CAF_high')
                    csv_out = final_df[out_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download scores CSV",
                        data=csv_out,
                        file_name="somics_scores.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.info(
                "MTX mode: ensure matrix.mtx, features.tsv, barcodes.tsv, and "
                "tissue_positions.csv are from the same 10x Visium run.\n\n"
                "CSV mode: ensure the expression CSV has spots as rows and "
                "Ensembl gene IDs as columns."
            )

    if 'live_results' in st.session_state:
        if st.button("Clear Results"):
            for key in ['live_results', 'live_model_type', 'live_image_bytes',
                        'live_image_name', 'live_scale_factor',
                        'live_spot_size', 'live_spot_opacity']:
                st.session_state.pop(key, None)
            st.rerun()

# ==========================================
# 9. PAGE: DOCUMENTATION
# ==========================================
elif page == "Documentation":
    st.markdown('<div class="main-header">Documentation</div>', unsafe_allow_html=True)

    doc_tabs = st.tabs(["Overview", "Model Architecture", "GUI User Guide"])

    with doc_tabs[0]:
        st.markdown(OVERVIEW)

    with doc_tabs[1]:
        st.markdown(MODEL_ARCH)

    with doc_tabs[2]:
        st.markdown(GUI_GUIDE)
