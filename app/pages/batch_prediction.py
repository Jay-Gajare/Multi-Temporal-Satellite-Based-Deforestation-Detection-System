"""Batch Prediction — folder upload, progress tracking, live logs, results table."""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.cards import metric_card
from app.components.layout import (
    page_header, section_header, divider, footer, empty_state, info_banner,
)
from app.utils.inference_wrapper import load_predictor


def _group_files_by_patch(files: list) -> dict[str, list]:
    """Group uploaded files by patch ID based on filename patterns.

    Expected: patch_XXXXXX_month_XX.tif or patch_XXXXXX/month_XX.tif
    Falls back to grouping by directory structure or using filename as patch ID.
    """
    groups = defaultdict(list)
    for f in files:
        name = f.name.lower()
        patch_id = None

        if "patch_" in name:
            parts = name.split("patch_")
            if len(parts) > 1:
                id_part = parts[1].split("_")[0].split("/")[0].split(".")[0]
                patch_id = f"patch_{id_part}"

        if not patch_id:
            patch_id = Path(f.name).stem.rsplit("_", 1)[0] if "_" in f.name else Path(f.name).stem

        groups[patch_id].append(f)

    return dict(groups)


def _group_files_by_folder(files: list) -> dict[str, list]:
    """Group files by their parent folder path."""
    groups = defaultdict(list)
    for f in files:
        parts = f.name.replace("\\", "/").split("/")
        if len(parts) > 1:
            folder = parts[0]
        else:
            folder = Path(f.name).stem.rsplit("_", 1)[0] if "_" in f.name else Path(f.name).stem
        groups[folder].append(f)
    return dict(groups)


def render() -> None:
    page_header(
        "Batch Prediction",
        "Run inference on multiple patches simultaneously",
        icon="📂",
        icon_gradient="var(--grad-cool)",
    )

    threshold = st.session_state.get("sidebar_threshold", 0.5)

    # ── Upload Zone ───────────────────────────────────────────────
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone__icon">📂</div>
        <div class="upload-zone__title">Upload patch directory</div>
        <div class="upload-zone__sub">
            Upload multiple GeoTIFF files. Files will be grouped by patch ID.<br>
            Supported: <code>patch_XXXXXX_month_XX.tif</code> or folder structures with <code>month_XX.tif</code> files.
        </div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload files",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_upload_v2",
    )

    if uploaded:
        groups = _group_files_by_patch(uploaded)
        n_patches = len(groups)
        n_files = len(uploaded)

        info_banner(
            f"{n_files} file(s) uploaded → grouped into **{n_patches}** patch(es)",
            icon="📎",
        )

        if n_patches > 100:
            st.warning(f"Processing {n_patches} patches may take a while on CPU (~{n_patches * 2.6 / 60:.0f} min estimated).")

        # ── Grouping Preview ──────────────────────────────────────
        with st.expander(f"📁 Patch Groups ({n_patches} patches)", expanded=False):
            for pid, files in list(groups.items())[:20]:
                file_names = [f.name for f in sorted(files, key=lambda x: x.name)]
                st.markdown(
                    f'<div style="display:flex;gap:8px;align-items:center;margin-bottom:4px;">'
                    f'<span style="font-size:12px;font-weight:700;color:var(--accent);min-width:140px;">{pid}</span>'
                    f'<span style="font-size:11px;color:var(--text-muted);">{len(files)} files: {", ".join(file_names[:3])}{"..." if len(files) > 3 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if n_patches > 20:
                st.caption(f"… and {n_patches - 20} more patches")

        # ── Run Button ────────────────────────────────────────────
        if st.button("▶ Run Batch Prediction", key="run_batch_v2", type="primary",
                     use_container_width=True):
            _run_batch(groups, threshold)
    else:
        empty_state(
            icon="📂",
            title="No files uploaded yet",
            description=(
                "Upload GeoTIFF files for batch deforestation detection.\n\n"
                "**Supported formats:**\n"
                "- `patch_XXXXXX_month_01.tif` … `month_12.tif`\n"
                "- Folder structures: `patch_001/month_01.tif` …\n"
                "- Any TIF files (will be grouped by name prefix)"
            ),
        )

    # ── Show Previous Results ─────────────────────────────────────
    if "batch_results_v2" in st.session_state and st.session_state.batch_results_v2:
        _render_results(st.session_state.batch_results_v2)

    footer()


def _run_batch(groups: dict[str, list], threshold: float) -> None:
    """Execute batch inference with progress, logs, and results collection."""
    predictor = load_predictor(threshold)
    n = len(groups)
    results = []
    errors = []

    progress_bar = st.progress(0, text="Initializing batch inference...")
    log_area = st.empty()
    stats_area = st.empty()
    start_time = time.perf_counter()

    log_lines = []
    n_defor = 0
    n_safe = 0

    for i, (patch_id, files) in enumerate(groups.items()):
        progress = (i) / n
        elapsed = time.perf_counter() - start_time
        eta = (elapsed / max(i, 1)) * (n - i) if i > 0 else 0

        progress_bar.progress(
            progress,
            text=f"Processing **{patch_id}** ({i+1}/{n}) — ETA: {eta:.0f}s",
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="dew_batch_"))
        try:
            for f in files:
                dest = tmp_dir / Path(f.name).name
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())

            t0 = time.perf_counter()
            result = predictor.predict(tmp_dir)
            latency = time.perf_counter() - t0

            result["patch_id"] = patch_id
            result["filename"] = files[0].name if files else patch_id
            result["n_files"] = len(files)
            result["batch_latency_s"] = latency

            pred_label = "DEFORESTATION" if result.get("prediction") == 1 else "SAFE"
            pred_color = "#EF4444" if result.get("prediction") == 1 else "#00C896"
            prob = result.get("probability", 0)

            if result.get("prediction") == 1:
                n_defor += 1
            else:
                n_safe += 1

            log_line = (
                f'<div style="display:flex;gap:8px;align-items:center;padding:3px 0;font-size:12px;">'
                f'<span style="color:var(--text-muted);font-family:var(--font-mono);min-width:28px;">{i+1:3d}</span>'
                f'<span style="color:var(--accent);font-weight:600;min-width:130px;">{patch_id}</span>'
                f'<span style="color:{pred_color};font-weight:700;">{pred_label}</span>'
                f'<span style="color:var(--text-muted);font-family:var(--font-mono);">P={prob:.4f}</span>'
                f'<span style="color:var(--text-muted);font-family:var(--font-mono);">{latency*1000:.0f}ms</span>'
                f'</div>'
            )
            log_lines.append(log_line)

            results.append(result)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:80]}"
            errors.append({"patch_id": patch_id, "error": error_msg})
            log_lines.append(
                f'<div style="display:flex;gap:8px;align-items:center;padding:3px 0;font-size:12px;">'
                f'<span style="color:var(--text-muted);font-family:var(--font-mono);min-width:28px;">{i+1:3d}</span>'
                f'<span style="color:var(--accent);font-weight:600;min-width:130px;">{patch_id}</span>'
                f'<span style="color:#EF4444;font-weight:700;">ERROR</span>'
                f'<span style="color:#EF4444;font-size:11px;">{error_msg}</span>'
                f'</div>'
            )
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if (i + 1) % 5 == 0 or (i + 1) == n:
            total_ms = (time.perf_counter() - start_time) * 1000
            avg_ms = total_ms / max(i + 1, 1)
            stats_html = (
                f'<div style="display:flex;gap:24px;padding:8px 12px;background:var(--bg-secondary);'
                f'border:1px solid var(--border);border-radius:8px;margin:8px 0;font-size:12px;">'
                f'<span>✅ <strong>{i+1}</strong>/{n}</span>'
                f'<span style="color:#EF4444;">🔴 {n_defor} deforestation</span>'
                f'<span style="color:#00C896;">🟢 {n_safe} safe</span>'
                f'<span>⏱️ <strong>{avg_ms:.0f}ms</strong> avg</span>'
                f'</div>'
            )
            stats_area.markdown(stats_html, unsafe_allow_html=True)
            log_area.markdown("".join(log_lines[-50:]), unsafe_allow_html=True)

    progress_bar.progress(1.0, text=f"✅ Complete — {n} patches processed")
    total_time = time.perf_counter() - start_time

    st.success(
        f"Batch complete: **{n}** patches in **{total_time:.1f}s** "
        f"({n_defor} deforestation, {n_safe} safe, {len(errors)} errors)"
    )

    st.session_state.batch_results_v2 = {
        "results": results,
        "errors": errors,
        "total_time": total_time,
        "threshold": threshold,
    }

    if results:
        _render_results(st.session_state.batch_results_v2)


def _render_results(data: dict) -> None:
    """Render the batch prediction results table and download buttons."""
    results = data.get("results", [])
    errors = data.get("errors", [])
    total_time = data.get("total_time", 0)
    threshold = data.get("threshold", 0.5)

    if not results:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Results Summary", f"{len(results)} successful · {len(errors)} failed · {total_time:.1f}s total", "📊")

    # ── Summary Metrics ───────────────────────────────────────────
    n_defor = sum(1 for r in results if r.get("prediction") == 1)
    n_safe = sum(1 for r in results if r.get("prediction") == 0)
    avg_time = np.mean([r.get("batch_latency_s", 0) for r in results]) * 1000
    avg_prob = np.mean([r.get("probability", 0) for r in results])
    avg_conf = np.mean([r.get("confidence", 0) for r in results])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Total Patches", str(len(results)), icon="📦", gradient="var(--grad-accent)")
    with c2:
        metric_card("Deforestation", str(n_defor), icon="🔴", gradient="var(--grad-warm)")
    with c3:
        metric_card("Safe", str(n_safe), icon="🟢", gradient="var(--grad-safe)")
    with c4:
        metric_card("Avg Latency", f"{avg_time:.0f}ms", icon="⏱️", gradient="var(--grad-cool)")
    with c5:
        metric_card("Avg Probability", f"{avg_prob:.3f}", icon="📊", gradient="var(--grad-primary)")

    # ── Interactive Results Table ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Detailed Results", "Click column headers to sort", "📋")

    rows = []
    for r in results:
        pred = r.get("prediction", 0)
        prob = r.get("probability", 0)
        conf = r.get("confidence", 0)
        latency_ms = r.get("batch_latency_s", 0) * 1000

        if pred == 1:
            pred_label = "🔴 Deforestation"
        else:
            pred_label = "🟢 No Deforestation"

        conf_label = "High" if conf >= 0.8 else "Medium" if conf >= 0.5 else "Low"

        rows.append({
            "Patch ID": r.get("patch_id", ""),
            "Prediction": pred_label,
            "Probability": round(prob, 4),
            "Confidence": round(conf, 4),
            "Conf. Level": conf_label,
            "Latency (ms)": round(latency_ms, 1),
            "Threshold": threshold,
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, 40 + len(rows) * 35),
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", min_value=0, max_value=1, format="%.4f",
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.4f",
            ),
            "Latency (ms)": st.column_config.NumberColumn(
                "Latency (ms)", format="%.1f",
            ),
        },
    )

    # ── Downloads ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Export Results", "", "📥")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_buf.getvalue(),
            file_name="batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        json_export = []
        for r in results:
            row = {k: v for k, v in r.items()
                   if not isinstance(v, np.ndarray) and k != "temporal_stack"}
            json_export.append(row)
        st.download_button(
            label="📋 Download JSON",
            data=json.dumps(json_export, indent=2, default=str),
            file_name="batch_predictions.json",
            mime="application/json",
            use_container_width=True,
        )

    with c3:
        summary = {
            "total_patches": len(results),
            "deforestation_count": n_defor,
            "safe_count": n_safe,
            "avg_latency_ms": round(avg_time, 1),
            "avg_probability": round(avg_prob, 4),
            "threshold": threshold,
            "total_time_s": round(total_time, 2),
            "errors": len(errors),
        }
        st.download_button(
            label="📊 Download Summary",
            data=json.dumps(summary, indent=2),
            file_name="batch_summary.json",
            mime="application/json",
            use_container_width=True,
        )

    if errors:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(f"⚠️ {len(errors)} Failed Patches", expanded=False):
            err_df = pd.DataFrame(errors)
            st.dataframe(err_df, use_container_width=True)
