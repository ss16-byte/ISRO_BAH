import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TARGETS = {
    "TOI-700 d (Earth-size Habitable)": {
        "tic": "TIC 150428135",
        "host_star": "TOI-700",
        "known_period": 37.42,
        "star_radius_solar": 0.42,
        "planet_family": "Earth-like",
        "description": "A temperate Earth-size planet around a quiet red dwarf.",
    },
    "L 98-59 d (Super-Earth)": {
        "tic": "TIC 307210830",
        "host_star": "L 98-59",
        "known_period": 3.6904,
        "star_radius_solar": 0.31,
        "planet_family": "Super-Earth",
        "description": "A compact rocky planet orbiting a nearby red dwarf.",
    },
    "Pi Mensae c (Super-Earth)": {
        "tic": "TIC 261136679",
        "host_star": "Pi Mensae",
        "known_period": 6.27,
        "star_radius_solar": 1.10,
        "planet_family": "Super-Earth",
        "description": "A well-known super-Earth around a bright nearby star.",
    },
    "HD 21749 b (Sub-Neptune)": {
        "tic": "TIC 279741379",
        "host_star": "HD 21749",
        "known_period": 35.57,
        "star_radius_solar": 0.70,
        "planet_family": "Mini-Neptune",
        "description": "A larger-than-Earth world useful as a TESS benchmark.",
    },
    "TRAPPIST-1 e (Earth-size)": {
        "tic": "TIC 278892590",
        "host_star": "TRAPPIST-1",
        "known_period": 6.10,
        "star_radius_solar": 0.12,
        "planet_family": "Earth-like",
        "description": "One of the famous TRAPPIST-1 planets, ideal for transit validation.",
    },
    "TOI-270 c (Mini-Neptune)": {
        "tic": "TIC 259377017",
        "host_star": "TOI-270",
        "known_period": 5.66,
        "star_radius_solar": 0.38,
        "planet_family": "Mini-Neptune",
        "description": "A compact multi-planet system useful for transit validation.",
    },
    "AU Mic b (Neptune-like)": {
        "tic": "TIC 441420236",
        "host_star": "AU Microscopii",
        "known_period": 8.46,
        "star_radius_solar": 0.75,
        "planet_family": "Neptune-like",
        "description": "A young planet orbiting an active nearby star.",
    },
    "WASP-18 b (Hot Jupiter)": {
        "tic": "TIC 100100827",
        "host_star": "WASP-18",
        "known_period": 0.9414,
        "star_radius_solar": 1.23,
        "planet_family": "Gas Giant",
        "description": "An ultra-short-period hot Jupiter benchmark.",
    },
    "WASP-121 b (Hot Jupiter)": {
        "tic": "TIC 22529346",
        "host_star": "WASP-121",
        "known_period": 1.2749,
        "star_radius_solar": 1.46,
        "planet_family": "Gas Giant",
        "description": "A famous hot Jupiter with a strong transit signature.",
    },
    "KELT-9 b (Ultra-Hot Jupiter)": {
        "tic": "TIC 16740101",
        "host_star": "KELT-9",
        "known_period": 1.48,
        "star_radius_solar": 2.36,
        "planet_family": "Gas Giant",
        "description": "An extreme ultra-hot Jupiter used as a stress-test target.",
    },
    "WASP-46 b (Hot Jupiter)": {
        "tic": "TIC 231663901",
        "host_star": "WASP-46",
        "known_period": 1.43,
        "star_radius_solar": 0.90,
        "planet_family": "Gas Giant",
        "description": "A hot Jupiter with a prominent transit signature.",
    },
}

PLANET_CLASSES = {
    "Earth-like": "A terrestrial rocky planet with a radius comparable to Earth.",
    "Super-Earth": "Larger than Earth but smaller than Neptune.",
    "Mini-Neptune": "A planet with a thick atmosphere, smaller than Neptune.",
    "Neptune-like": "A larger gaseous planet with a substantial atmosphere.",
    "Gas Giant": "A massive planet similar to Jupiter or Saturn.",
}

PLANET_COLORS = {
    "Earth-like": ("#60a5fa", "#1d4ed8"),
    "Super-Earth": ("#7c3aed", "#4c1d95"),
    "Mini-Neptune": ("#22c55e", "#166534"),
    "Neptune-like": ("#06b6d4", "#0f766e"),
    "Gas Giant": ("#f59e0b", "#b45309"),
}


# ============================================================
# HELPERS
# ============================================================

def to_float(value, default=np.nan):
    try:
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value).reshape(-1)[0]
        return float(value)
    except Exception:
        return default


def classify_planet(radius_earth: float) -> str:
    if radius_earth < 1.25:
        return "Earth-like"
    if radius_earth < 2.0:
        return "Super-Earth"
    if radius_earth < 4.0:
        return "Mini-Neptune"
    if radius_earth < 10.0:
        return "Neptune-like"
    return "Gas Giant"


def confidence_score(peak_power: float, noise_reduction_pct: float = 0.0, period_error_pct: float | None = None) -> float:
    """
    Heuristic demo score. Not a formal statistical confidence interval.
    """
    score = 50.0

    # BLS peak contribution
    score += (peak_power / (peak_power + 15.0)) * 25.0

    # Noise reduction contribution
    score += np.clip(noise_reduction_pct, 0.0, 100.0) * 0.15

    # Period accuracy contribution if we know the benchmark period
    if period_error_pct is not None and np.isfinite(period_error_pct):
        score += max(0.0, 18.0 - (period_error_pct * 3.0))

    return float(np.clip(score, 50.0, 95.0))


def draw_planet_doodle(planet_class: str, title: str):
    base_color, accent_color = PLANET_COLORS.get(planet_class, ("#60a5fa", "#1d4ed8"))

    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    # Glow background
    grid = np.linspace(-1.0, 1.0, 300)
    X, Y = np.meshgrid(grid, grid)
    R = np.sqrt(X**2 + Y**2)
    glow = np.clip(1.0 - R, 0.0, 1.0)
    ax.imshow(glow, extent=(-1, 1, -1, 1), origin="lower", cmap="Blues", alpha=0.35)

    # Planet body
    planet = Circle((0, 0), 0.6, facecolor=base_color, edgecolor="#e2e8f0", linewidth=2.5, alpha=0.98)
    ax.add_patch(planet)

    # Shading
    shade = Circle((-0.13, 0.12), 0.52, facecolor="white", edgecolor="none", alpha=0.08)
    ax.add_patch(shade)

    # Highlight
    highlight = Circle((0.18, 0.20), 0.12, facecolor="white", edgecolor="none", alpha=0.18)
    ax.add_patch(highlight)

    # Ring for large planets
    if planet_class in {"Gas Giant", "Neptune-like"}:
        ring = Ellipse((0, 0), width=1.7, height=0.42, angle=-20, fill=False, edgecolor=accent_color, linewidth=3, alpha=0.7)
        ax.add_patch(ring)

    # Tiny stars
    ax.scatter([-0.95, 0.88, -0.7, 0.75], [0.82, 0.70, -0.68, -0.75], s=[10, 8, 6, 5], c=["#f8fafc"] * 4, alpha=0.8)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    ax.set_title(title, color="#f8fafc", fontsize=11, pad=12)
    plt.tight_layout()
    return fig


def run_bls_search(lightcurve, period_min, period_max, period_steps=9000):
    periods = np.linspace(period_min, period_max, period_steps)

    # Transit duration must be shorter than the shortest period in the scan
    duration_max = min(0.2, max(0.02, period_min * 0.35))
    durations = np.linspace(0.01, duration_max, 12)

    bls = lightcurve.to_periodogram(
        method="bls",
        period=periods,
        duration=durations,
    )

    best_period = bls.period_at_max_power
    best_t0 = bls.transit_time_at_max_power
    best_duration = bls.duration_at_max_power
    best_depth = bls.depth_at_max_power
    peak_power = float(np.nanmax(bls.power.value))

    return {
        "periodogram": bls,
        "best_period": best_period,
        "best_t0": best_t0,
        "best_duration": best_duration,
        "best_depth": best_depth,
        "peak_power": peak_power,
        "period_value": to_float(best_period),
        "duration_hours": to_float(best_duration.to("h")),
        "depth_value": to_float(best_depth),
    }


def analyze_target(target_meta: dict):
    try:
        tic_id = target_meta["tic"]
        known_period = target_meta.get("known_period", None)
        star_radius_solar = target_meta.get("star_radius_solar", 1.0)

        # Download
        search_result = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC")
        if len(search_result) == 0:
            search_result = lk.search_lightcurve(tic_id, mission="TESS")

        if len(search_result) == 0:
            return {"error": f"No TESS light curve found for {tic_id}."}

        lc = search_result[0].download(flux_column="pdcsap_flux")
        if lc is None:
            return {"error": "Failed to download a valid light curve."}

        # Normalize / preprocess
        lc = lc.remove_nans()
        lc_raw = lc.normalize()
        raw_noise = float(np.nanstd(lc_raw.flux.value))

        npts = len(lc_raw.flux)
        window_length = 101 if npts >= 101 else max(5, (npts // 2) * 2 + 1)

        # Lightweight denoising placeholder
        lc_clean = lc_raw.remove_outliers(sigma=5.0).flatten(window_length=window_length)
        denoised_noise = float(np.nanstd(lc_clean.flux.value))

        noise_reduction_pct = (
            ((raw_noise - denoised_noise) / raw_noise) * 100.0
            if raw_noise > 0
            else 0.0
        )

        # Search range
        if known_period is not None:
            period_min = max(0.15, known_period * 0.5)
            period_max = min(50.0, known_period * 1.5)
        else:
            period_min, period_max = 0.5, 10.0

        if period_max <= period_min:
            period_max = period_min + 0.5

        period_steps = 9000 if (period_max - period_min) <= 10 else 6000

        # Raw and denoised BLS
        raw_bls = run_bls_search(lc_raw, period_min, period_max, period_steps)
        denoised_bls = run_bls_search(lc_clean, period_min, period_max, period_steps)

        # Final detection uses denoised data
        best_period = denoised_bls["best_period"]
        best_t0 = denoised_bls["best_t0"]
        best_duration = denoised_bls["best_duration"]
        best_depth = denoised_bls["best_depth"]

        # Phase folding
        try:
            lc_folded = lc_clean.fold(period=best_period, epoch_time=best_t0)
        except Exception:
            lc_folded = lc_clean.fold(period=best_period)

        # Radius estimate
        depth_for_radius = max(0.0, denoised_bls["depth_value"])
        r_planet_earth = star_radius_solar * np.sqrt(depth_for_radius) * 109.2

        # Classification
        planet_class = classify_planet(r_planet_earth)
        planet_description = PLANET_CLASSES[planet_class]

        # Period errors
        raw_period_error_pct = None
        denoised_period_error_pct = None
        if known_period is not None and known_period > 0:
            raw_period_error_pct = abs(raw_bls["period_value"] - known_period) / known_period * 100.0
            denoised_period_error_pct = abs(denoised_bls["period_value"] - known_period) / known_period * 100.0

        # Confidence
        raw_confidence = confidence_score(raw_bls["peak_power"], 0.0, raw_period_error_pct)
        denoised_confidence = confidence_score(
            denoised_bls["peak_power"],
            noise_reduction_pct,
            denoised_period_error_pct,
        )

        # Quality score
        data_quality_score = float(
            np.clip(
                100.0
                - (denoised_noise * 25000.0)
                + (noise_reduction_pct * 0.4),
                0.0,
                100.0,
            )
        )

        # Approx transit count
        try:
            time_span_days = float(np.nanmax(lc_clean.time.value) - np.nanmin(lc_clean.time.value))
            approx_transits = max(1, int(np.floor(time_span_days / denoised_bls["period_value"]))) if denoised_bls["period_value"] > 0 else 1
        except Exception:
            approx_transits = 1

        return {
            "success": True,
            "target_id": tic_id,
            "target_name": target_meta.get("label", tic_id),
            "host_star": target_meta.get("host_star", ""),
            "known_period": known_period,
            "star_radius_solar": star_radius_solar,
            "planet_class": planet_class,
            "planet_description": planet_description,
            "planet_family": target_meta.get("planet_family", planet_class),

            "period": denoised_bls["period_value"],
            "epoch": str(best_t0),
            "duration_hours": denoised_bls["duration_hours"],
            "depth": denoised_bls["depth_value"],
            "r_planet_earth": r_planet_earth,

            "raw_noise": raw_noise,
            "denoised_noise": denoised_noise,
            "noise_reduction_pct": noise_reduction_pct,

            "raw_period": raw_bls["period_value"],
            "denoised_period": denoised_bls["period_value"],
            "raw_period_error_pct": raw_period_error_pct,
            "denoised_period_error_pct": denoised_period_error_pct,

            "raw_confidence": raw_confidence,
            "detection_confidence": denoised_confidence,
            "confidence_delta": denoised_confidence - raw_confidence,

            "data_quality_score": data_quality_score,
            "approx_transits": approx_transits,

            "lc_raw": lc_raw,
            "lc_clean": lc_clean,
            "raw_periodogram": raw_bls["periodogram"],
            "bls_periodogram": denoised_bls["periodogram"],
            "lc_folded": lc_folded,
        }

    except Exception as e:
        return {"error": f"Astrophysics pipeline execution failure: {str(e)}"}


def fmt_num(value, decimals=4, suffix=""):
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return "N/A"
        return f"{value:.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def build_impact_table(results):
    rows = []

    rows.append({
        "Metric": "Flux Noise (Std Dev)",
        "Raw": fmt_num(results["raw_noise"], 6),
        "Denoised": fmt_num(results["denoised_noise"], 6),
        "Change": fmt_num(results["raw_noise"] - results["denoised_noise"], 6),
    })

    if results.get("raw_period_error_pct") is not None and results.get("denoised_period_error_pct") is not None:
        rows.append({
            "Metric": "Detected Period Error vs Known (%)",
            "Raw": fmt_num(results["raw_period_error_pct"], 3),
            "Denoised": fmt_num(results["denoised_period_error_pct"], 3),
            "Change": fmt_num(results["raw_period_error_pct"] - results["denoised_period_error_pct"], 3),
        })
    else:
        rows.append({
            "Metric": "Detected Period Error vs Known (%)",
            "Raw": "N/A",
            "Denoised": "N/A",
            "Change": "N/A",
        })

    rows.append({
        "Metric": "Detection Confidence (%)",
        "Raw": fmt_num(results["raw_confidence"], 1),
        "Denoised": fmt_num(results["detection_confidence"], 1),
        "Change": fmt_num(results["confidence_delta"], 1),
    })

    rows.append({
        "Metric": "Estimated Radius (R⊕)",
        "Raw": "N/A",
        "Denoised": fmt_num(results["r_planet_earth"], 2),
        "Change": "N/A",
    })

    return pd.DataFrame(rows)


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Antyrix Dashboard",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.1rem; }
            section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌌 Antyrix: AI-Assisted Exoplanet Detection")
    st.markdown("**First-year ISRO Hackathon Project — Powered by AI Denoising & Physics Validation**")
    st.divider()

    # Sidebar
    st.sidebar.header("Input Parameters")
    selected_name = st.sidebar.selectbox("Select Exoplanet", list(TARGETS.keys()))
    target_meta = TARGETS[selected_name]
    target_tic = target_meta["tic"]

    st.sidebar.info(f"**Target ID:** `{target_tic}`")
    st.sidebar.caption("Use the dropdown to pick a benchmark system. Judges remember planet names faster than TIC IDs.")

    st.sidebar.markdown("### About selected target")
    st.sidebar.write(f"**Host star:** {target_meta['host_star']}")
    st.sidebar.write(f"**Planet family:** {target_meta['planet_family']}")
    st.sidebar.write(f"**Known period:** {target_meta['known_period']:.4f} days")
    st.sidebar.caption(target_meta["description"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pipeline Steps")
    st.sidebar.markdown(
        "1. Downloading TESS data  \n"
        "2. Signal denoising  \n"
        "3. BLS search  \n"
        "4. Validation  \n"
        "5. Classification"
    )

    run_button = st.sidebar.button("Run Detection Pipeline", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About Antyrix")
    st.sidebar.caption(
        "Antyrix combines signal conditioning with physics-based BLS transit detection "
        "to keep the pipeline interpretable while improving signal quality."
    )

    if run_button:
        with st.spinner(f"Downloading and analyzing {selected_name}..."):
            results = analyze_target(target_meta)

        if "error" in results:
            st.error(results["error"])
            return

        st.success(f"Analysis complete for {selected_name} ({target_tic})!")

        # Top metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Orbital Period", f"{results['period']:.4f} days")
        c2.metric("Transit Depth", f"{results['depth'] * 100:.4f} %")
        c3.metric("Transit Duration", f"{results['duration_hours']:.2f} hours")
        c4.metric(
            "Estimated Radius",
            f"{results['r_planet_earth']:.2f} R⊕",
            help=f"Approx. stellar radius used: {results['star_radius_solar']:.2f} R☉",
        )
        c5.metric("Detection Confidence", f"{results['detection_confidence']:.1f} %")

        # Classification + planet doodle
        st.markdown("### 🪐 Planet Classification")
        p_left, p_right = st.columns([1.0, 1.5])

        with p_left:
            planet_fig = draw_planet_doodle(results["planet_class"], f"{results['planet_class']} World")
            st.pyplot(planet_fig, use_container_width=True)
            plt.close(planet_fig)

        with p_right:
            st.success(results["planet_class"])
            st.write(results["planet_description"])
            st.caption(f"Family: {results['planet_family']}")
            st.write(f"**Host Star:** {results['host_star']}")
            st.write(f"**Approx. transits observed:** {results['approx_transits']}")
            st.write(f"**Data Quality Score:** {results['data_quality_score']:.1f} / 100")
            if results["known_period"] is not None:
                st.write(f"**Known Period:** {results['known_period']:.5f} days")
                st.write(f"**Detected Period Error:** {results['denoised_period_error_pct']:.3f} %")

        # AI impact
        st.markdown("### 🧠 Signal Conditioning Impact")
        st.caption("The denoising stage currently uses a lightweight flattening placeholder. Replace this step later with the learned denoiser without changing the rest of the UI.")
        impact_df = build_impact_table(results)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

        if results["noise_reduction_pct"] >= 0:
            st.success(
                f"Signal conditioning reduced flux noise by {results['noise_reduction_pct']:.2f}% "
                f"and the detection confidence is {results['detection_confidence']:.1f}%."
            )
        else:
            st.warning(
                f"Signal conditioning increased flux noise slightly ({results['noise_reduction_pct']:.2f}%). "
                "Check the periodogram and validation tab."
            )

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Denoised Light Curve",
                "📊 BLS Periodogram",
                "⚛️ Phase-Folded Curve",
                "🛡️ Validation Summary",
            ]
        )

        with tab1:
            st.subheader("1. Denoised Light Curve")
            fig1, ax1 = plt.subplots(figsize=(11, 4))
            results["lc_clean"].plot(ax=ax1, color="black", lw=0.5)
            ax1.set_title(f"Denoised Light Curve — {selected_name}")
            st.pyplot(fig1)
            plt.close(fig1)
            st.caption(
                "This is the cleaned flux after outlier removal and detrending. "
                "In the final pipeline, this slot can be replaced by the learned denoiser."
            )

        with tab2:
            st.subheader("2. Box Least Squares Periodogram")
            fig2, ax2 = plt.subplots(figsize=(11, 4))
            results["bls_periodogram"].plot(ax=ax2, color="blue")
            ax2.axvline(
                results["period"],
                color="red",
                linestyle="--",
                label=f"Peak: {results['period']:.4f} d",
            )
            ax2.legend()
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption("The tallest peak marks the strongest repeating transit-like signal candidate.")

        with tab3:
            st.subheader("3. Phase-Folded Transit Curve")
            fig3, ax3 = plt.subplots(figsize=(11, 4))
            results["lc_folded"].plot(ax=ax3, color="purple", alpha=0.25, label="All points")
            try:
                results["lc_folded"].bin(time_bin_size=0.01).plot(
                    ax=ax3,
                    color="red",
                    marker="o",
                    ls="",
                    label="Binned data",
                )
            except Exception:
                pass
            ax3.legend()
            st.pyplot(fig3)
            plt.close(fig3)
            st.caption("Phase folding stacks multiple orbits together so the transit dip becomes easier to see.")

        with tab4:
            st.subheader("4. Validation Summary")

            v1, v2 = st.columns(2)

            with v1:
                st.markdown("#### Core Metrics")
                st.write(f"**Target:** `{selected_name}`")
                st.write(f"**TIC ID:** `{target_tic}`")
                st.write(f"**Detected Period:** `{results['period']:.5f} days`")
                st.write(f"**Transit Depth:** `{results['depth'] * 100:.5f} %`")
                st.write(f"**Transit Duration:** `{results['duration_hours']:.3f} hours`")
                st.write(f"**Estimated Radius:** `{results['r_planet_earth']:.3f} Earth radii`")
                st.write(f"**Detection Confidence:** `{results['detection_confidence']:.1f} %`")
                st.write(f"**Data Quality Score:** `{results['data_quality_score']:.1f} / 100`")

                if results["known_period"] is not None:
                    st.write(f"**Known Period:** `{results['known_period']:.5f} days`")
                    st.write(f"**Period Error:** `{results['denoised_period_error_pct']:.3f} %`")
                else:
                    st.write("**Known Period:** `N/A`")

            with v2:
                st.markdown("#### Planet Description")
                st.success(results["planet_class"])
                st.write(results["planet_description"])
                st.markdown("#### Validation Status")
                st.progress(int(results["detection_confidence"]))
                st.caption(
                    f"Confidence meter based on BLS peak strength + signal conditioning impact. "
                    f"Approx. transits observed: {results['approx_transits']}."
                )

                if results["known_period"] is not None:
                    if results["denoised_period_error_pct"] <= 1.0:
                        st.success("The detected period is very close to the known benchmark period.")
                    elif results["denoised_period_error_pct"] <= 5.0:
                        st.info("The detected period is reasonably close to the benchmark period.")
                    else:
                        st.warning("The detected period is not very close to the benchmark period. Check the periodogram.")

        st.divider()
        st.caption(
            "Note: the denoising stage is currently a placeholder using flattening. "
            "The rest of the dashboard is ready to keep the same structure when a learned denoiser is plugged in."
        )

    else:
        st.info("Choose a target from the dropdown and click **Run Detection Pipeline**.")


if __name__ == "__main__":
    main()