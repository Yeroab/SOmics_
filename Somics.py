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
from somics_inference_adapted import run_inference_from_bytes, log1p_cpm

# ==========================================
# 0. DOWNLOAD DEMO DATA (if using external hosting)
# ==========================================
import os
from pathlib import Path

@st.cache_resource
def ensure_demo_files():
    demo_dir = Path("demo_zips")
    demo_dir.mkdir(exist_ok=True)
    DEMO_URLS = {}
    return True

# ==========================================
# 1. PAGE SETUP & THEME
# ==========================================
st.set_page_config(page_title="SOmics: CAF-Immune", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #40E0D0; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #20B2AA; text-align: center; margin-bottom: 2rem; }
    .stMetric { background-color: #E0F7FA; padding: 15px; border-radius: 10px; border-left: 5px solid #40E0D0; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #E0F7FA 0%, #B2EBF2 100%); }
    .stMetric label { font-size: 12px !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 12px !important; }
    .stMetric [data-testid="stMetricDelta"] { display: none !important; }
    [data-testid="stFileUploader"] { max-height: 120px; }
    [data-testid="stFileUploader"] section {
        padding: 0.5rem;
        border: 2px dashed #20B2AA !important;
        border-radius: 8px;
        background-color: #F0FFFF;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #008B8B !important;
        background-color: #E0F7FA;
    }
    [data-testid="stFileUploader"] section > div { min-height: 80px; max-height: 80px; }
    [data-testid="stFileUploader"] small { display: none !important; }
    [data-testid="stFileUploader"] span { color: #20B2AA !important; font-size: 0.85rem !important; display: inline !important; }
    [data-testid="stFileUploader"] p { color: #20B2AA !important; font-size: 0.85rem !important; }
    [data-testid="stFileUploader"] button {
        display: block !important;
        background-color: #20B2AA !important;
        color: white !important;
        border: 1px solid #20B2AA !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
        border-radius: 5px;
        margin-top: 0.3rem;
    }
    [data-testid="stFileUploader"] button span { display: inline !important; color: white !important; }
    [data-testid="stFileUploader"] button:hover {
        background-color: #008B8B !important;
        border: 1px solid #008B8B !important;
    }
    [data-testid="stFileUploader"] > div > div { padding: 0.5rem; text-align: center; }
    [data-testid="stFileUploader"] svg {
        width: 2.5rem !important;
        height: 2.5rem !important;
        display: block !important;
        margin: 0 auto 0.3rem auto !important;
        color: #20B2AA !important;
    }
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
# ==========================================

def run_inference_csv(df, model, features):
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
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        arr = tifffile.imread(io.BytesIO(raw))
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[1]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        return Image.open(io.BytesIO(raw))


def overlay_spots_on_image(pil_image, final_df, scale_factor=1.0, spot_opacity=0.85, spot_size=8):
    img_w, img_h = pil_image.size
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG', optimize=False, compress_level=1)  # minimal compression = sharper
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    x_coords = final_df['pxl_col'].values * scale_factor
    y_coords = final_df['pxl_row'].values * scale_factor
    fig = go.Figure()
    fig.add_layout_image(
        source=b64,
        x=0, y=0,
        xref="x", yref="y",
        sizex=img_w, sizey=img_h,
        xanchor="left", yanchor="top",
        layer="below",
        opacity=1.0
    )
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            color=final_df['Score'].values,
            colorscale=[[0, "#FF6B6B"], [0.5, "#FFFFFF"], [1, "#40E0D0"]],
            cmin=0, cmax=1,
            size=spot_size,
            opacity=spot_opacity,
            colorbar=dict(title="Immune Score", thickness=20, len=0.75),
            line=dict(width=0),
        ),
        text=final_df['barcode'].values,
        hovertemplate="<b>%{text}</b><br>Score: %{marker.color:.3f}<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(
        xaxis=dict(range=[0, img_w], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[img_h, 0], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=900,
        title=dict(
            text="CAF-Immune Spatial Map — Tissue Overlay",
            font=dict(size=20),
        ),
        plot_bgcolor="black",
    )
    return fig


def parse_positions(pos_file_bytes, filename):
    try:
        pos = pd.read_csv(io.BytesIO(pos_file_bytes))
        if pos.columns[0].startswith('Unnamed') or pos.columns[0] not in [
                'barcode', 'Barcode', 'barcodes']:
            raise ValueError("No header detected")
    except Exception:
        pos = pd.read_csv(io.BytesIO(pos_file_bytes), header=None)
        pos.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
    bc_candidates = ['barcode', 'Barcode', 'barcodes', 'spot_id']
    bc_col = next((c for c in bc_candidates if c in pos.columns), pos.columns[0])
    if bc_col != 'barcode':
        pos = pos.rename(columns={bc_col: 'barcode'})
    if 'pxl_row_in_fullres' in pos.columns:
        pos = pos.rename(columns={
            'pxl_row_in_fullres': 'pxl_row',
            'pxl_col_in_fullres': 'pxl_col'
        })
    if 'in_tissue' not in pos.columns:
        pos['in_tissue'] = 1
    return pos


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## SOmics")
    page = st.radio("Go to:", ["Home", "Demo Walkthrough", "Classify - User Analysis", "Documentation"])

# ==========================================
# 6. PAGE: HOME
# ==========================================
if page == "Home":
    st.markdown('<div class="main-header">SOmics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Spatial Analysis of the CAF-Immune Axis</div>',
                unsafe_allow_html=True)
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
    This demo runs the full SOmics pipeline on a real ovarian cancer spatial
    transcriptomics sample. All files are bundled with the app — no upload required.
    """)

    if not assets_loaded:
        st.error("Model assets could not be loaded. Cannot run demo.")
        st.stop()

    import zipfile
    import tempfile

    DEMO_SAMPLES = {
        "308": {"name": "OCS Sample 308"},
        "308_hgsc": {"name": "HGSC Sample 308"},
        "309": {"name": "OCS Sample 309"},
        "309_hgsc": {"name": "HGSC Sample 309"},
        "310": {"name": "OCS Sample 310"},
        "310_hgsc": {"name": "HGSC Sample 310"},
        "311": {"name": "OCS Sample 311"},
        "311_hgsc": {"name": "HGSC Sample 311"},
    }

    @st.cache_data
    def load_demo_results(_rf_model, _lr_model, _model_features, sample_id, model_type="Random Forest"):
        import os
        DEMO_DIR = f"demo_zips/{sample_id}"

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
        img = Image.open(os.path.join(DEMO_DIR, "tissue_lowres_image.png"))
        model = _rf_model if model_type == "Random Forest" else _lr_model
        final_df = run_inference_from_bytes(raw_mtx, raw_feat, raw_bc, pos_df, model, _model_features)
        return final_df, img, scale_factor

    col_select1, col_select2 = st.columns(2)
    with col_select1:
        sample_names = {k: v["name"] for k, v in DEMO_SAMPLES.items()}
        selected_sample = st.selectbox(
            "Select Sample",
            options=list(sample_names.keys()),
            format_func=lambda x: sample_names[x]
        )
    with col_select2:
        demo_model = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True)

    if st.button("Run Demo Analysis", type="primary"):
        with st.spinner(f"Running pipeline on {DEMO_SAMPLES[selected_sample]['name']}..."):
            try:
                final_df, demo_img, scale_factor = load_demo_results(
                    rf_model, lr_model, model_features, selected_sample, demo_model
                )
                st.session_state.demo_results     = final_df
                st.session_state.demo_img         = demo_img
                st.session_state.demo_scale       = scale_factor
                st.session_state.demo_model_used  = demo_model
                st.session_state.demo_sample_used = selected_sample
                st.success(f"Demo analysis complete. Analyzed {len(final_df)} spots from {DEMO_SAMPLES[selected_sample]['name']}.")
                st.rerun()
            except FileNotFoundError as e:
                st.error(
                    f"Demo data files not found: {e}\n\n"
                    f"Ensure the `demo_zips/` folder contains subfolders:\n"
                    f"308/, 308_hgsc/, 309/, 309_hgsc/, 310/, 310_hgsc/, 311/, 311_hgsc/\n\n"
                    f"Each folder should contain: matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz, "
                    f"tissue_positions_list.csv, tissue_lowres_image.png, and scalefactors_json.json."
                )
            except Exception as e:
                st.error(f"Error running demo: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    if 'demo_results' in st.session_state:
        final_df     = st.session_state.demo_results
        demo_img     = st.session_state.demo_img
        scale_factor = st.session_state.demo_scale

        st.success(f"Displaying results for {len(final_df)} tissue spots")

        # ── High-resolution scatter ──────────────────────────────────
        fig = px.scatter(
            final_df, x='pxl_col', y='pxl_row', color='Score',
            color_continuous_scale=["#FF6B6B", "#FFFFFF", "#40E0D0"],
            title=f"CAF-Immune Spatial Map ({st.session_state.demo_model_used})",
            labels={'Score': 'Immune Score', 'pxl_col': 'X Position', 'pxl_row': 'Y Position'},
            height=900,
            width=1200,
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(
            font=dict(size=14),
            title_font_size=20,
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        sample_name = DEMO_SAMPLES[st.session_state.demo_sample_used]['name']
        st.caption(
            f"{sample_name} — {len(final_df)} in-tissue spots  |  "
            f"Model: {st.session_state.demo_model_used}"
        )

        st.divider()
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Total Spots", len(final_df))
        with col_d2:
            immune_n = (final_df['Score'] > 0.5).sum()
            st.metric("Immune-high", immune_n)
        with col_d3:
            caf_n = (final_df['Score'] <= 0.5).sum()
            st.metric("CAF-high", caf_n)
        with col_d4:
            st.metric("Mean Score", f"{final_df['Score'].mean():.3f}")

        st.divider()
        col_hist, col_info = st.columns([2, 1])
        with col_hist:
            # ── High-resolution histogram ────────────────────────────
            fig_hist = px.histogram(
                final_df, x='Score', nbins=40,
                color_discrete_sequence=["#40E0D0"],
                title="Distribution of CAF-Immune Scores Across Spots",
                labels={'Score': 'Immune Score (0=CAF-high, 1=Immune-high)'},
                height=500,
                width=900,
            )
            fig_hist.update_layout(
                font=dict(size=13),
                title_font_size=18,
                bargap=0.05,
            )
            fig_hist.add_vline(
                x=0.5, line_dash="dash", line_color="gray",
                annotation_text="Threshold", annotation_font_size=13
            )
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
            for k in ['demo_results', 'demo_img', 'demo_scale', 'demo_model_used', 'demo_sample_used']:
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

    # ========== EXAMPLE ANALYSIS SECTION ==========
    with st.expander("📊 Try Example Analysis - HGSC Sample 308", expanded=False):
        st.info("""
        Run analysis on a real High-Grade Serous Ovarian Cancer (HGSC) spatial transcriptomics sample.
        This demonstrates how SOmics classifies tissue spots along the CAF-Immune axis.
        """)

        col_ex1, col_ex2 = st.columns([1, 3])
        with col_ex1:
            example_model = st.selectbox("Model", ["Random Forest", "Logistic Regression"], key="example_model_select")
            if st.button("Run Example", type="primary", key="run_example"):
                with st.spinner("Running analysis..."):
                    try:
                        import gzip
                        import os

                        possible_paths = [
                            '',
                            'user-data/',
                            '/mount/src/somics_/user-data/',
                        ]

                        data_path = None
                        for path in possible_paths:
                            test_file = os.path.join(path, 'HGSC_308_coordinates_for_CARD.csv')
                            if os.path.exists(test_file):
                                data_path = path
                                break

                        if data_path is None:
                            st.error("Example data files not found. Please ensure these files are in your repo:\n- barcodes_308__3__tsv.gz\n- features_308_tsv.gz\n- matrix__2__mtx.gz\n- HGSC_308_coordinates_for_CARD.csv")
                            st.stop()

                        with gzip.open(os.path.join(data_path, 'barcodes 308 (3).tsv.gz'), 'rb') as f:
                            raw_bc = f.read()
                        with gzip.open(os.path.join(data_path, 'features 308.tsv.gz'), 'rb') as f:
                            raw_feat = f.read()
                        with gzip.open(os.path.join(data_path, 'matrix (2).mtx.gz'), 'rb') as f:
                            raw_mtx = f.read()

                        pos_df = pd.read_csv(os.path.join(data_path, 'HGSC_308_coordinates_for_CARD.csv'))
                        pos_df = pos_df.rename(columns={
                            'x': 'pxl_col',
                            'y': 'pxl_row',
                            'Spot_ID': 'barcode'
                        })
                        if 'in_tissue' not in pos_df.columns:
                            pos_df['in_tissue'] = 1
                        if 'array_row' not in pos_df.columns:
                            pos_df['array_row'] = 0
                        if 'array_col' not in pos_df.columns:
                            pos_df['array_col'] = 0

                        active_model = rf_model if example_model == "Random Forest" else lr_model
                        final_df = run_inference_from_bytes(raw_mtx, raw_feat, raw_bc, pos_df, active_model, model_features)

                        st.session_state.example_results = final_df
                        st.session_state.example_model_type = example_model

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        with st.expander("Show detailed error trace"):
                            st.code(traceback.format_exc())

                        with st.expander("Debug Information"):
                            st.write("Checking file locations...")
                            import os
                            for path in ['', 'user-data/', '/mount/src/somics_/user-data/']:
                                st.write(f"\nChecking path: '{path}'")
                                try:
                                    files = os.listdir(path if path else '.')
                                    st.write(f"Files found: {[f for f in files if 'HGSC' in f or 'barcode' in f or 'feature' in f or 'matrix' in f]}")
                                except:
                                    st.write("Path does not exist")

        with col_ex2:
            if 'example_results' in st.session_state:
                final_df = st.session_state.example_results

                # ── High-resolution example scatter ──────────────────
                fig = px.scatter(
                    final_df, x='pxl_col', y='pxl_row', color='Score',
                    color_continuous_scale=["#FF6B6B", "#FFFFFF", "#40E0D0"],
                    title=f"HGSC 308 - {st.session_state.example_model_type}",
                    labels={'Score': 'Immune Score'},
                    height=600,
                    width=900,
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(font=dict(size=13), title_font_size=16)
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Spots", len(final_df))
                with col_m2:
                    immune_n = (final_df['Score'] > 0.5).sum()
                    st.metric("Immune-high", f"{immune_n/len(final_df):.1%}")
                with col_m3:
                    csv_out = final_df[['barcode', 'Score', 'pxl_row', 'pxl_col']].to_csv(index=False).encode('utf-8')
                    st.download_button("Download", csv_out, "hgsc_308.csv", "text/csv")

    st.markdown("---")
    st.markdown("### Upload Your Data")

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

    # Model selector - full width
    model_type = st.selectbox("Model", ["Random Forest", "Logistic Regression"])

    # ==========================================
    # UPLOAD SECTION
    # ==========================================
    if input_mode == "MTX (raw 10x Visium)":
        st.markdown("### Upload 10x Visium Files")

        # ── Row 1: expression files ──────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Barcodes File**")
            bc_upload = st.file_uploader(
                "barcodes.tsv or barcodes.tsv.gz",
                type=['tsv', 'gz'],
                key="bc_upload",
                label_visibility="collapsed",
            )
            bc_file = bc_upload if bc_upload and 'barcode' in bc_upload.name.lower() else None
            if bc_upload:
                st.success(f"✓ {bc_upload.name}") if bc_file else st.warning("Expected a barcodes file")

        with col2:
            st.markdown("**Features File**")
            feat_upload = st.file_uploader(
                "features.tsv or features.tsv.gz",
                type=['tsv', 'gz'],
                key="feat_upload",
                label_visibility="collapsed",
            )
            feat_file = feat_upload if feat_upload and any(
                k in feat_upload.name.lower() for k in ('feature', 'gene')) else None
            if feat_upload:
                st.success(f"✓ {feat_upload.name}") if feat_file else st.warning("Expected a features/genes file")

        with col3:
            st.markdown("**Matrix File**")
            mtx_file = st.file_uploader(
                "matrix.mtx or matrix.mtx.gz",
                type=['mtx', 'gz'],
                label_visibility="collapsed",
            )

        # ── Spacer ───────────────────────────────────────────────────
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        # ── Row 2: spatial / optional files ─────────────────────────
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("**Tissue Positions**")
            pos_file = st.file_uploader(
                "tissue_positions.csv (or _list.csv)",
                type=['csv'],
                label_visibility="collapsed",
            )

        with col5:
            st.markdown("**Tissue Image (optional)**")
            image_file = st.file_uploader(
                "Tissue image",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                label_visibility="collapsed",
            )

        with col6:
            st.markdown("**Scale Factors (optional)**")
            sf_file = st.file_uploader(
                "scalefactors_json.json",
                type=['json'],
                label_visibility="collapsed",
            )

        barcode_feature_files = [f for f in [bc_upload, feat_upload] if f is not None]
        expr_file = None

    else:  # CSV mode
        st.markdown("### Upload CSV Files")

        # ── Row 1 ────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Expression CSV**")
            expr_file = st.file_uploader(
                "Expression CSV",
                type=['csv'],
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("**Tissue Positions CSV**")
            pos_file = st.file_uploader(
                "Tissue positions CSV",
                type=['csv'],
                label_visibility="collapsed",
            )

        with col3:
            st.markdown("**Tissue Image (optional)**")
            image_file = st.file_uploader(
                "Tissue image (optional)",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                label_visibility="collapsed",
            )

        # ── Spacer ───────────────────────────────────────────────────
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

        # ── Row 2: scale factors alone ───────────────────────────────
        col4, _, _ = st.columns(3)
        with col4:
            st.markdown("**Scale Factors (optional)**")
            sf_file = st.file_uploader(
                "scalefactors_json.json (optional)",
                type=['json'],
                label_visibility="collapsed",
            )

        mtx_file = feat_file = bc_file = None

    # ==========================================
    # SCALE FACTOR / IMAGE DISPLAY SETTINGS
    # ==========================================
    if image_file is not None:
        st.markdown("### Image Display Settings")
        if sf_file is not None:
            st.info("Scale factor will be read from scalefactors_json.json")
            scale_factor = None
            col1, col2 = st.columns(2)
            with col1:
                spot_size = st.slider("Spot size", 3, 20, 8)
            with col2:
                spot_opacity = st.slider("Spot opacity", 0.1, 1.0, 0.85, 0.05)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                image_res = st.radio("Image resolution", ["Full-resolution", "Downsampled"])
            with col2:
                if image_res == "Downsampled":
                    scale_factor = st.number_input("Scale factor", 0.001, 1.0, 0.05, 0.001, format="%.3f")
                else:
                    scale_factor = 1.0
            with col3:
                spot_size = st.slider("Spot size", 3, 20, 8)
                spot_opacity = st.slider("Spot opacity", 0.1, 1.0, 0.85, 0.05)
    else:
        scale_factor = 1.0
        spot_size = 8
        spot_opacity = 0.85

    # ==========================================
    # RUN PREDICTION
    # ==========================================
    mtx_ready = input_mode == "MTX (raw 10x Visium)" and all(f is not None for f in [mtx_file, feat_file, bc_file, pos_file])
    csv_ready = input_mode == "CSV (pre-converted)" and all(f is not None for f in [expr_file, pos_file])

    if mtx_ready or csv_ready:
        try:
            active_model = rf_model if model_type == "Random Forest" else lr_model

            if st.button("Run Prediction", type="primary"):
                with st.spinner("Running inference..."):
                    pos_bytes = pos_file.read()
                    pos_df = parse_positions(pos_bytes, pos_file.name)

                    if input_mode == "MTX (raw 10x Visium)":
                        raw_mtx = mtx_file.read()
                        raw_feat = feat_file.read()
                        raw_bc = bc_file.read()

                        if mtx_file.name.endswith('.gz'):
                            raw_mtx = gzip.decompress(raw_mtx)
                        if feat_file.name.endswith('.gz'):
                            raw_feat = gzip.decompress(raw_feat)
                        if bc_file.name.endswith('.gz'):
                            raw_bc = gzip.decompress(raw_bc)

                        final_df = run_inference_from_bytes(raw_mtx, raw_feat, raw_bc, pos_df, active_model, model_features)
                        if 'pCAF' in final_df.columns and 'Score' not in final_df.columns:
                            final_df = final_df.rename(columns={'pCAF': 'Score'})
                    else:
                        expr_file.seek(0)
                        df_expr = pd.read_csv(expr_file, index_col=0)
                        scores = run_inference_csv(df_expr, active_model, model_features)
                        results = pd.DataFrame({'barcode': df_expr.index, 'Score': scores})
                        final_df = pd.merge(results, pos_df, on='barcode')

                    resolved_scale = scale_factor
                    if sf_file is not None:
                        sf_file.seek(0)
                        sf_data = json.load(sf_file)
                        if image_file is not None and 'hires' in image_file.name.lower():
                            resolved_scale = sf_data.get('tissue_hires_scalef', 1.0)
                        else:
                            resolved_scale = sf_data.get('tissue_lowres_scalef', 0.05)

                    st.session_state.live_results = final_df
                    st.session_state.live_model_type = model_type
                    st.session_state.live_scale_factor = resolved_scale
                    st.session_state.live_spot_size = spot_size
                    st.session_state.live_spot_opacity = spot_opacity

                    if image_file is not None:
                        image_file.seek(0)
                        st.session_state.live_image_bytes = image_file.read()
                        st.session_state.live_image_name = image_file.name
                    else:
                        for k in ['live_image_bytes', 'live_image_name']:
                            st.session_state.pop(k, None)

            if 'live_results' in st.session_state:
                st.markdown("---")
                st.markdown("### Results")

                final_df = st.session_state.live_results

                if 'live_image_bytes' in st.session_state:
                    show_tissue = st.checkbox("Show tissue image overlay", value=False)

                    if show_tissue:
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
                        st.caption(f"Image: {img_w} x {img_h} px | Scale: {st.session_state.live_scale_factor} | {len(final_df)} spots | Model: {st.session_state.live_model_type}")
                    else:
                        st.info("Check the box above to view tissue image with spot overlay")
                else:
                    # ── High-resolution user analysis scatter ────────
                    fig = px.scatter(
                        final_df, x='pxl_col', y='pxl_row', color='Score',
                        color_continuous_scale=["#FF6B6B", "#FFFFFF", "#40E0D0"],
                        title=f"CAF-Immune Spatial Map ({st.session_state.live_model_type})",
                        labels={'Score': 'Immune Score', 'pxl_col': 'X', 'pxl_row': 'Y'},
                        height=900,
                        width=1200,
                    )
                    fig.update_traces(marker=dict(size=6))
                    fig.update_layout(font=dict(size=14), title_font_size=20)
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Total Spots", len(final_df))
                with col_r2:
                    immune_n = (final_df['Score'] > 0.5).sum()
                    st.metric("Immune-high Spots", f"{immune_n} ({immune_n/len(final_df):.1%})")
                with col_r3:
                    st.metric("Mean Score", f"{final_df['Score'].mean():.3f}")

                with st.expander("Download Results"):
                    out_cols = ['barcode', 'Score']
                    if 'CAF_high' in final_df.columns:
                        out_cols.append('CAF_high')
                    csv_out = final_df[out_cols].to_csv(index=False).encode('utf-8')
                    st.download_button("Download scores CSV", csv_out, "somics_scores.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("MTX mode: ensure all files are from the same 10x Visium run.\nCSV mode: ensure expression CSV has spots as rows and Ensembl gene IDs as columns.")

    if 'live_results' in st.session_state:
        if st.button("Clear Results", key="clear_user_upload"):
            for key in ['live_results', 'live_model_type', 'live_image_bytes', 'live_image_name', 'live_scale_factor', 'live_spot_size', 'live_spot_opacity']:
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
