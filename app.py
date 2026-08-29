# -*- coding: utf-8 -*-
"""
Zephrock -- browser GUI (Streamlit), matching the Zephmatic/Zephslide/Zephflac
pattern. Wraps the unchanged zephrock.HoekBrownMaterial core -- no new
analysis logic lives in this file, only an interface.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from zephrock import HoekBrownMaterial, PowerCurveCriterion, BartonBandisJoint

st.set_page_config(page_title="Zephrock", page_icon="\U0001faa8", layout="wide")

st.title("Zephrock")
st.caption("Rock-mass and joint strength criteria — a free, from-scratch alternative "
           "to the relevant core of Rocscience's RSData/RocData/RocLab.")

criterion = st.radio(
    "Criterion", ["Generalized Hoek-Brown → Mohr-Coulomb", "Power curve (fit to test data)",
                  "Barton-Bandis (joint shear strength)"],
    horizontal=True)

if criterion == "Power curve (fit to test data)":
    st.header("Power curve — nonlinear shear-strength envelope fit to your own data")
    st.caption("tau = sigci · A · (sigma_n / sigci) ^ B, fit by least squares to shear-test "
               "points you enter — an alternative to the GSI-based approach, for when you "
               "have real direct-shear or triaxial (converted) data instead of a field GSI estimate.")
    pc_sigci = st.number_input("Reference stress, sigci (MPa)", min_value=0.01, value=50.0, step=1.0,
                                key="pc_sigci")
    st.write("Enter shear-test data points (sigma_n, tau), one pair per row:")
    default_df = pd.DataFrame({"sigma_n_MPa": [0.5, 1.0, 2.0, 5.0, 10.0],
                                "tau_MPa": [0.6, 1.0, 1.6, 3.0, 5.0]})
    data_df = st.data_editor(default_df, num_rows="dynamic", key="pc_data")

    valid = data_df.dropna()
    if len(valid) >= 2:
        try:
            pc = PowerCurveCriterion.fit_from_shear_data(
                valid["sigma_n_MPa"].values, valid["tau_MPa"].values, pc_sigci)
            r2 = pc.r_squared(valid["sigma_n_MPa"].values, valid["tau_MPa"].values)
            col1, col2, col3 = st.columns(3)
            col1.metric("A", f"{pc.A:.4f}")
            col2.metric("B", f"{pc.B:.4f}")
            col3.metric("R² (log-log fit)", f"{r2:.4f}")

            sn_eval = st.number_input("Evaluate instantaneous c'/phi' at sigma_n (MPa)",
                                       min_value=0.01, value=float(valid["sigma_n_MPa"].median()))
            c_i, phi_i = pc.instantaneous_params(sn_eval)
            st.write(f"At sigma_n={sn_eval:.3f} MPa: instantaneous **c'={c_i*1000:.1f} kPa**, "
                     f"**phi'={phi_i:.2f} deg** (tangent to the fitted curve at this point).")

            sn_range = np.linspace(valid["sigma_n_MPa"].min(), valid["sigma_n_MPa"].max(), 200)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(valid["sigma_n_MPa"], valid["tau_MPa"], label="Input data", zorder=3)
            ax.plot(sn_range, pc.shear_strength(sn_range), "--", label="Fitted power curve")
            ax.set_xlabel("sigma_n (MPa)")
            ax.set_ylabel("tau (MPa)")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        except ValueError as e:
            st.error(str(e))
    else:
        st.info("Enter at least 2 valid (sigma_n>0, tau>0) data points to fit a curve.")

    with st.expander("About this criterion"):
        st.markdown("""
This is a standard nonlinear power-law shear-strength envelope, fit directly to
data you supply — not derived from RSData's own internal power-curve algorithm
(that formulation was not independently verified against a reliable source
before building this, so this is offered as an honest, useful nonlinear-fit
tool in its own right, not a claimed exact replication). Instantaneous
cohesion/friction at any normal stress come from the exact analytic
derivative of the fitted curve, cross-checked against numerical
differentiation during development.
""")

elif criterion == "Barton-Bandis (joint shear strength)":
    st.header("Barton-Bandis — peak shear strength of a single rock joint")
    st.caption("tau = sigma_n · tan[ JRC · log10(JCS / sigma_n) + phi_r ]  "
               "(Barton & Choubey 1977; Barton & Bandis 1990) — a different physical "
               "quantity from rock-mass strength above: this is joint/discontinuity "
               "shear strength, the input Zephmatic-style wedge/planar checks need.")
    bcol1, bcol2, bcol3 = st.columns(3)
    JRC = bcol1.number_input("JRC (Joint Roughness Coefficient, 0-20)", min_value=0.0, max_value=20.0,
                              value=10.0, step=0.5)
    JCS = bcol2.number_input("JCS (Joint wall Compressive Strength, MPa)", min_value=0.1, value=80.0,
                              step=1.0)
    phi_r = bcol3.number_input("Residual friction angle, phi_r (deg)", min_value=0.0, max_value=70.0,
                                value=30.0, step=0.5)
    bb = BartonBandisJoint(JRC=JRC, JCS=JCS, phi_r=phi_r)

    sigma_n_eval = st.number_input("Effective normal stress, sigma_n (MPa)", min_value=0.01,
                                    value=min(2.0, JCS * 0.5), step=0.1)
    if sigma_n_eval >= JCS:
        st.warning(f"sigma_n ({sigma_n_eval:.2f} MPa) ≥ JCS ({JCS:.2f} MPa) — outside this "
                   "criterion's valid range; result below should not be trusted.")
    else:
        tau = bb.shear_strength(sigma_n_eval)
        phi_sec = bb.secant_friction_angle_deg(sigma_n_eval)
        c_i, phi_i = bb.instantaneous_params(sigma_n_eval)
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak shear strength, tau", f"{tau * 1000:.1f} kPa")
        col2.metric("Secant friction angle", f"{phi_sec:.2f} deg")
        col3.metric("Instantaneous phi_i / c_i", f"{phi_i:.2f} deg / {c_i * 1000:.1f} kPa")

        sn_max = min(JCS * 0.98, sigma_n_eval * 3 + 1)
        sn_range = np.linspace(0.01, sn_max, 200)
        tau_range = [bb.shear_strength(s) for s in sn_range]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sn_range, tau_range, label="Barton-Bandis envelope")
        ax.scatter([sigma_n_eval], [tau], color="red", zorder=3, label="Evaluation point")
        ax.set_xlabel("sigma_n (MPa)")
        ax.set_ylabel("tau (MPa)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with st.expander("About this criterion"):
        st.markdown("""
Standard Barton-Bandis joint shear-strength model, unchanged from the
original publications. JRC and JCS are field/lab-estimated inputs, not
computed by this tool — JRC from comb-profile comparison charts or a tilt
test, JCS from a Schmidt hammer rebound test (equals intact UCS for an
unweathered joint). Only valid for sigma_n < JCS.
""")

else:
    # ------------------------------------------------------- sidebar (HB) --
    st.sidebar.header("1. Rock mass inputs (Hoek-Brown)")
    sigci = st.sidebar.number_input("Intact UCS, sigci (MPa)", min_value=0.01, value=69.04, step=1.0)
    mi = st.sidebar.number_input("Intact material constant, mi", min_value=0.1, value=9.0, step=0.5,
                                  help="Hoek's mi tables: limestone ~8-12, dolomite ~9-10.1, marble ~9.")
    GSI = st.sidebar.slider("Geological Strength Index, GSI", min_value=1, max_value=100, value=70,
                             help="Field-estimated rock-mass quality rating (Hoek & Marinos 2000 chart) "
                                  "-- NOT computed by this tool.")
    D = st.sidebar.slider("Disturbance factor, D", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                           help="0 = undisturbed/TBM, 1 = heavily blast-disturbed open-pit production blasting.")

    hb = HoekBrownMaterial(sigci=sigci, mi=mi, GSI=GSI, D=D)

    st.sidebar.header("2. Confining stress range (sigma3max)")
    s3_mode = st.sidebar.radio("Convention", ["Slope (Eq. 19)", "Tunnel (Eq. 18)", "Custom value"], index=0)

    if s3_mode == "Slope (Eq. 19)":
        gamma = st.sidebar.number_input("Unit weight (kN/m3)", min_value=1.0, value=26.5, step=0.5)
        height = st.sidebar.number_input("Slope height, H (m)", min_value=0.1, value=10.0, step=1.0)
        sigma3max = hb.sigma3max_slope(gamma / 1000.0, height)
    elif s3_mode == "Tunnel (Eq. 18)":
        gamma = st.sidebar.number_input("Unit weight (kN/m3)", min_value=1.0, value=26.5, step=0.5)
        depth = st.sidebar.number_input("Tunnel depth, H (m)", min_value=0.1, value=50.0, step=5.0)
        sigma3max = hb.sigma3max_tunnel(gamma / 1000.0, depth)
    else:
        sigma3max = st.sidebar.number_input("sigma3max (MPa)", min_value=0.001, value=0.25 * sigci, step=0.5)

    st.sidebar.caption(f"sigma3max = **{sigma3max:.3f} MPa**  (sigma3n = {sigma3max / sigci:.4f})")

    # --------------------------------------------------------- main panel --
    tab_results, tab_plot, tab_plugin, tab_about = st.tabs(
        ["Results", "Envelope chart", "Plug into Zephslide", "About / validation"])

    mc = hb.equivalent_mohr_coulomb(sigma3max)

    with tab_results:
        col1, col2, col3 = st.columns(3)
        col1.metric("mb", f"{hb.mb:.4f}")
        col1.metric("s", f"{hb.s:.6f}")
        col1.metric("a", f"{hb.a:.4f}")
        col2.metric("Rock mass UCS, sigma_c", f"{hb.sigma_c:.2f} MPa")
        col2.metric("Tensile strength, sigma_t", f"{hb.sigma_t:.3f} MPa")
        col2.metric("Global strength, sigma_cm", f"{hb.sigma_cm:.2f} MPa")
        col3.metric("Cohesion, c'", f"{mc.cohesion_MPa * 1000:.1f} kPa")
        col3.metric("Friction angle, phi'", f"{mc.phi_deg:.2f} deg")
        col3.metric("Deformation modulus (est.)", f"{hb.deformation_modulus_GPa():.2f} GPa")

        st.divider()
        st.write(mc.text_summary())

        summary_txt = (
            f"Zephrock output\n"
            f"Inputs: sigci={sigci} MPa, mi={mi}, GSI={GSI}, D={D}\n"
            f"sigma3max convention: {s3_mode}, sigma3max={sigma3max:.4f} MPa\n\n"
            f"mb={hb.mb:.6f}, s={hb.s:.8f}, a={hb.a:.6f}\n"
            f"sigma_c={hb.sigma_c:.4f} MPa, sigma_t={hb.sigma_t:.4f} MPa, sigma_cm={hb.sigma_cm:.4f} MPa\n"
            f"Equivalent Mohr-Coulomb: c'={mc.cohesion_MPa * 1000:.2f} kPa, phi'={mc.phi_deg:.3f} deg\n"
            f"Deformation modulus (estimate): {hb.deformation_modulus_GPa():.3f} GPa\n"
        )
        st.download_button("Download results (.txt)", summary_txt, file_name="zephrock_results.txt")

    with tab_plot:
        s3 = np.linspace(1e-9, sigma3max, 300)
        s1 = hb.sigma1(s3)
        # Equivalent MC line: sigma1 = sigma3*tan^2(45+phi/2) + 2c*tan(45+phi/2)
        phi_rad = np.radians(mc.phi_deg)
        k = np.tan(np.pi / 4 + phi_rad / 2) ** 2
        s1_mc = s3 * k + 2 * mc.cohesion_MPa * np.sqrt(k)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(s3, s1, label="Hoek-Brown (actual criterion)", linewidth=2)
        ax.plot(s3, s1_mc, "--", label="Equivalent Mohr-Coulomb (fit)", linewidth=2)
        ax.set_xlabel("sigma3' (MPa)")
        ax.set_ylabel("sigma1' (MPa)")
        ax.set_title("Hoek-Brown envelope vs. equivalent Mohr-Coulomb fit")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button("Download chart (.png)", buf.getvalue(), file_name="zephrock_envelope.png")

    with tab_plugin:
        st.write("Hand this rock mass's equivalent Mohr-Coulomb parameters straight to "
                 "**Zephslide**'s circular-search limit-equilibrium solver -- no manual "
                 "re-typing of c'/phi'.")
        try:
            from zeph_common.geometry import Bench
            from zephslide.geometry import Slope as LEMSlope
            from zephslide.search import search_critical_circle

            pc1, pc2, pc3 = st.columns(3)
            bench_height = pc1.number_input("Bench height (m)", min_value=0.1, value=10.0, step=1.0)
            face_angle = pc2.number_input("Face angle (deg)", min_value=1.0, max_value=89.0, value=80.0)
            unit_weight = pc3.number_input("Unit weight (kN/m3)", min_value=1.0, value=26.5, step=0.5,
                                            key="plugin_uw")

            if st.button("Run Zephslide with this Zephrock material"):
                bench = Bench(height=bench_height, face_angle=face_angle, berm_width=0.0)
                material = hb.to_material(sigma3max=sigma3max, unit_weight_kNm3=unit_weight,
                                           name="Zephrock-derived")
                slope = LEMSlope(benches=[bench], material=material, name="Zephrock-derived")
                with st.spinner("Searching for critical circle..."):
                    result = search_critical_circle(slope, n_slices=30, nx=12, ny=10, n_exit=18,
                                                     method="bishop", refine=True, verbose=False)
                if result is None:
                    st.warning("No valid critical circle found for this geometry/material combination.")
                else:
                    st.success(f"Bishop FoS = {result.fos_result.fos:.3f}")
                    st.write(f"Material used: c'={material.cohesion:.1f} kPa, phi'={material.phi:.2f} deg, "
                             f"unit weight={material.unit_weight} kN/m3")
                    st.write(f"Critical circle: center=({result.xc:.2f}, {result.yc:.2f}), R={result.R:.2f} m")
        except ImportError:
            st.info("Zephslide isn't bundled with this deployment, so this tab is inactive here. "
                    "The full toolkit (zeph_common + zephslide + zephrock together) enables it.")

    with tab_about:
        st.markdown("""
**Scope:** core generalized Hoek-Brown (Hoek, Carranza-Torres & Corkum, 2002) to
equivalent Mohr-Coulomb only -- one of three criteria this app now offers (see
the "Criterion" selector above for the power curve and Barton-Bandis options).

**Validation:** every closed-form equation used here (mb/s/a, rock-mass strength,
sigma3max for slopes/tunnels, equivalent c'/phi') was cross-checked
character-for-character against a peer-reviewed restatement (Geosciences
12(7):262, MDPI, 2022) and matches exactly. A supplementary self-consistency
check (independently sampling the Hoek-Brown curve and Mohr-Coulomb-fitting it
via the Balmer 1952 relations, without using the closed-form equations at all)
agrees on phi' to within a few percent; c' shows more spread, a known artifact
of that check's own least-squares weighting, not evidence of an error here.

**Honest limitation:** an attempt to reproduce this project's own literature
values (Saliu & Shehu, 2012, Obajana/Ewekoro RocLab output) across four
different sigma3max conventions did not find a combination matching both c'
and phi' simultaneously -- because the source paper never states which
sigma3max (or depth/height) it used. Use a sigma3max convention you can state
and defend for your own site (the slope-height-based Eq. 19 above is
recommended whenever a real bench height is known), rather than treating any
existing published number as a target to reverse-engineer.

**GSI and mi are not computed by this tool** -- GSI needs field mapping
(Hoek & Marinos 2000 chart) or a stated assumption; mi comes from Hoek's own
published material tables.
""")
