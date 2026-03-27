"""
AASHTO 1993 Flexible Pavement Design — 5 Layers
ออกแบบผิวทางลาดยาง ตามวิธี AASHTO 1993 (5 ชั้น)
กรมทางหลวง - Department of Highways Thailand
"""

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import brentq

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AASHTO 1993 | ออกแบบผิวทาง 5 ชั้น",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
h1,h2,h3 { font-family: 'Sarabun', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

.header-box {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 60%, #0a1628 100%);
    border: 1px solid #2d5a8a; border-radius: 12px;
    padding: 24px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(13,97,166,0.3);
}
.header-box h1 { color: #e6f3ff; font-size: 1.8rem; font-weight: 700; margin: 0 0 4px 0; }
.header-box p  { color: #7ab3d4; margin: 0; font-size: 0.95rem; }

.result-card {
    background: linear-gradient(135deg, #1a2e1a 0%, #0f1f0f 100%);
    border: 1px solid #2d6b2d; border-radius: 10px;
    padding: 18px; text-align: center;
}
.result-card .label { color: #7dd87d; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.result-card .value { color: #c8f5c8; font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.result-card .unit  { color: #7dd87d; font-size: 0.8rem; }

.result-card-blue {
    background: linear-gradient(135deg, #1a2a3e 0%, #0f1a2d 100%);
    border: 1px solid #2d5a8a; border-radius: 10px;
    padding: 18px; text-align: center;
}
.result-card-blue .label { color: #7ab3d4; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.result-card-blue .value { color: #b3d9f5; font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; }
.result-card-blue .unit  { color: #7ab3d4; font-size: 0.8rem; }

.section-header {
    background: linear-gradient(90deg, #1e3a5f 0%, transparent 100%);
    border-left: 4px solid #0d74e7;
    padding: 8px 14px; border-radius: 0 8px 8px 0; margin: 14px 0 10px 0;
}
.section-header h3 { color: #a8d0f0; margin: 0; font-size: 0.95rem; font-weight: 600; }

.layer-box {
    border-radius: 6px; padding: 10px 14px; margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    display: flex; justify-content: space-between; align-items: center;
}
.layer-wearing  { background: #1a1505; border: 1px solid #c8a020; color: #f0d060; }
.layer-binder   { background: #1a1200; border: 1px solid #8a7020; color: #d4b84a; }
.layer-base     { background: #0d1a0d; border: 1px solid #4a7a4a; color: #8ac88a; }
.layer-subbase  { background: #0f0f1a; border: 1px solid #4a4a8a; color: #8a8ac8; }
.layer-select   { background: #1a1015; border: 1px solid #7a4a6a; color: #c88ab8; }
.layer-subgrade { background: #1a0f0a; border: 1px solid #8a5a3a; color: #c8a07a; }

.formula-box {
    background: #0a0f1a; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; color: #7ab3d4; line-height: 2;
}
.info-box {
    background: #111d2e; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 16px; font-size: 0.85rem; color: #8aafcc; line-height: 1.7;
}

[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab-list"] { background: #111d2e; border-radius: 8px; padding: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── AASHTO 1993 Functions ────────────────────────────────────────────────────
ZR_TABLE = {
    50:-0.000, 60:-0.253, 70:-0.524, 75:-0.674, 80:-0.842,
    85:-1.037, 90:-1.282, 91:-1.341, 92:-1.405, 93:-1.476,
    94:-1.555, 95:-1.645, 96:-1.751, 97:-1.881, 98:-2.054,
    99:-2.327, 99.9:-3.090,
}

DRAIN_COEFF = {
    "ดีเยี่ยม (< 2 ชม.)":    {"< 1%":1.40,"1-5%":1.35,"5-25%":1.30,"> 25%":1.20},
    "ดี (2-24 ชม.)":          {"< 1%":1.35,"1-5%":1.25,"5-25%":1.15,"> 25%":1.00},
    "ปานกลาง (1-7 วัน)":     {"< 1%":1.25,"1-5%":1.15,"5-25%":1.05,"> 25%":0.80},
    "แย่ (1-4 สัปดาห์)":     {"< 1%":1.15,"1-5%":1.05,"5-25%":0.80,"> 25%":0.60},
    "แย่มาก (> 1 เดือน)":    {"< 1%":1.05,"1-5%":0.95,"5-25%":0.75,"> 25%":0.40},
}
DRAIN_KEYS = list(DRAIN_COEFF.keys())
SAT_KEYS   = ["< 1%","1-5%","5-25%","> 25%"]

def get_ZR(r):
    levels = sorted(ZR_TABLE.keys())
    if r <= levels[0]:  return ZR_TABLE[levels[0]]
    if r >= levels[-1]: return ZR_TABLE[levels[-1]]
    for i in range(len(levels)-1):
        if levels[i] <= r <= levels[i+1]:
            r1,r2 = levels[i],levels[i+1]
            return ZR_TABLE[r1] + (ZR_TABLE[r2]-ZR_TABLE[r1])*(r-r1)/(r2-r1)

def aashto_eq(SN, W18, ZR, S0, delta_psi, MR):
    log_W18 = math.log10(W18)
    term1 = ZR * S0
    term2 = 9.36 * math.log10(SN+1) - 0.20
    psi_ratio = delta_psi / (4.2 - 1.5)
    if psi_ratio <= 0: return float('inf')
    term3 = math.log10(psi_ratio) / (0.40 + 1094/((SN+1)**5.19))
    term4 = 2.32 * math.log10(MR) - 8.07
    return (term1 + term2 + term3 + term4) - log_W18

def solve_SN(W18, ZR, S0, delta_psi, MR):
    try:
        return brentq(aashto_eq, 0.01, 30.0,
                      args=(W18, ZR, S0, delta_psi, MR),
                      xtol=1e-6, maxiter=300)
    except Exception:
        return None

def cbr_to_MR(cbr): return 1500.0 * cbr


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 6px 0;">
        <div style="font-size:2.2rem;">🛣️</div>
        <div style="color:#7ab3d4;font-weight:700;font-size:1rem;letter-spacing:1px;">AASHTO 1993</div>
        <div style="color:#4a7a9b;font-size:0.75rem;">5-Layer Flexible Pavement</div>
    </div>
    <hr style="border-color:#1e3a5f;margin:8px 0;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ พารามิเตอร์การออกแบบ")

    st.markdown('<div class="section-header"><h3>🚛 ปริมาณจราจร (W18)</h3></div>', unsafe_allow_html=True)
    w18_mode = st.radio("วิธีกรอก", ["W18 โดยตรง","คำนวณจาก AADT"], horizontal=True)

    if w18_mode == "W18 โดยตรง":
        W18 = st.number_input("W18 (ESAL)", min_value=1e4, max_value=1e9, value=5e6, format="%.2e", step=1e5)
    else:
        AADT        = st.number_input("AADT (คัน/วัน)", 100, 200000, 8000, 500)
        T_pct       = st.number_input("% รถบรรทุก", 1.0, 60.0, 15.0, 1.0)
        design_life = st.number_input("อายุออกแบบ (ปี)", 5, 40, 20, 5)
        TF          = st.number_input("Truck Factor (LEF)", 0.1, 5.0, 0.50, 0.05)
        growth_rate = st.number_input("อัตราการเติบโต (%/ปี)", 0.0, 10.0, 3.0, 0.5)
        GF = ((1+growth_rate/100)**design_life - 1)/(growth_rate/100) if growth_rate > 0 else float(design_life)
        W18 = AADT*(T_pct/100)*365*GF*TF
        st.info(f"**W18 = {W18:,.0f} ESAL**")

    st.markdown('<div class="section-header"><h3>📊 ความน่าเชื่อถือ</h3></div>', unsafe_allow_html=True)
    R_preset = st.selectbox("ประเภททาง", [
        "ทางหลวงพิเศษ (R=99%)", "ทางหลวงระหว่างเมือง (R=95%)",
        "ทางหลวงสายหลัก (R=90%)", "ทางหลวงสายรอง (R=85%)",
        "ถนนในเมืองสายหลัก (R=95%)", "ถนนในเมืองสายรอง (R=80%)", "กำหนดเอง"])
    R_map = {
        "ทางหลวงพิเศษ (R=99%)":99, "ทางหลวงระหว่างเมือง (R=95%)":95,
        "ทางหลวงสายหลัก (R=90%)":90, "ทางหลวงสายรอง (R=85%)":85,
        "ถนนในเมืองสายหลัก (R=95%)":95, "ถนนในเมืองสายรอง (R=80%)":80, "กำหนดเอง":90
    }
    if R_preset == "กำหนดเอง":
        reliability = st.slider("Reliability (%)", 50, 99, 90)
    else:
        reliability = R_map[R_preset]
        st.info(f"R = {reliability}%")
    ZR = get_ZR(reliability)
    S0 = st.number_input("S0 (Overall Std. Dev.)", 0.30, 0.60, 0.45, 0.01)

    st.markdown('<div class="section-header"><h3>📉 Serviceability</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    pi = c1.number_input("pi (เริ่มต้น)", 3.5, 5.0, 4.2, 0.1)
    pt = c2.number_input("pt (สิ้นสุด)", 1.5, 3.5, 2.5, 0.1)
    delta_psi = pi - pt
    st.info(f"DPSI = {delta_psi:.1f}")

    st.markdown('<div class="section-header"><h3>🏔️ Subgrade MR</h3></div>', unsafe_allow_html=True)
    MR_method = st.radio("กรอก MR จาก", ["CBR (%)","MR (psi)","MR (MPa)"])
    if MR_method == "CBR (%)":
        CBR_val = st.number_input("CBR (%)", 1.0, 30.0, 5.0, 0.5)
        MR = cbr_to_MR(CBR_val)
        st.info(f"MR = {MR:,.0f} psi  ({MR/145.038:.1f} MPa)")
    elif MR_method == "MR (psi)":
        MR = float(st.number_input("MR (psi)", 1000, 30000, 7500, 250))
        CBR_val = MR/1500
    else:
        MR_mpa = st.number_input("MR (MPa)", 10, 200, 52, 5)
        MR = MR_mpa * 145.038
        CBR_val = MR/1500


# ─── Solve SN ────────────────────────────────────────────────────────────────
SN_req = solve_SN(W18, ZR, S0, delta_psi, MR)

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛣️ ออกแบบผิวทางลาดยาง 5 ชั้น — AASHTO 1993</h1>
    <p>กรมทางหลวง &nbsp;|&nbsp; Department of Highways Thailand &nbsp;|&nbsp;
       Wearing · Binder · Base · Subbase · Select Fill</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📐 ออกแบบโครงสร้างทาง","📊 Sensitivity Analysis","📋 ตารางค่าสัมประสิทธิ์","ℹ️ ทฤษฎีและสูตร"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="result-card">
            <div class="label">SN Required</div>
            <div class="value">{SN_req:.3f}</div>
            <div class="unit">Structural Number</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="result-card-blue">
            <div class="label">Design ESAL</div>
            <div class="value">{W18:,.0f}</div>
            <div class="unit">W18</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="result-card-blue">
            <div class="label">Reliability</div>
            <div class="value">{reliability}%</div>
            <div class="unit">ZR = {ZR:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="result-card-blue">
            <div class="label">Subgrade MR</div>
            <div class="value">{MR:,.0f}</div>
            <div class="unit">psi ({MR/145.038:.1f} MPa)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏗️ กำหนดวัสดุและค่าสัมประสิทธิ์ทั้ง 5 ชั้น")

    # ── Layer Definitions ─────────────────────────────────────────────────────
    LAYER_DEFS = [
        {
            "no":1,
            "label":"ชั้นที่ 1 — ผิวทางลาดยาง (Asphalt Concrete)",
            "css":"layer-wearing", "emoji":"🟡",
            "mat_opts":["ผิวทางลาดยาง AC (a=0.40)","Dense Graded Asphalt (a=0.44)",
                        "Polymer Modified Asphalt (a=0.46)","กำหนดเอง"],
            "mat_a":[0.40, 0.44, 0.46, 0.40],
            "D_min_cm":5, "D_max_cm":20,
            "a_def":0.40, "a_min":0.25, "a_max":0.55,
            "m_def":1.1,  "use_def":True,
        },
        {
            "no":2,
            "label":"ชั้นที่ 2 — พื้นทางซีเมนต์ (CTBAC — Cement Treated Base)",
            "css":"layer-binder", "emoji":"🟠",
            "mat_opts":["พื้นทางซีเมนต์ CTBAC (a=0.18)","Cement-treated Base (a=0.20)",
                        "Lean Concrete (a=0.23)","กำหนดเอง"],
            "mat_a":[0.18, 0.20, 0.23, 0.18],
            "D_min_cm":10, "D_max_cm":40,
            "a_def":0.18, "a_min":0.10, "a_max":0.35,
            "m_def":1.1,  "use_def":True,
        },
        {
            "no":3,
            "label":"ชั้นที่ 3 — พื้นทางหินคลุก 80% AC (Crushed Stone 80% AC)",
            "css":"layer-base", "emoji":"🟢",
            "mat_opts":["หินคลุก 80% AC (a=0.13)","Crushed Stone Base (a=0.14)",
                        "Dense Graded Gravel (a=0.12)","กำหนดเอง"],
            "mat_a":[0.13, 0.14, 0.12, 0.13],
            "D_min_cm":10, "D_max_cm":60,
            "a_def":0.13, "a_min":0.05, "a_max":0.40,
            "m_def":1.1,  "use_def":True,
        },
        {
            "no":4,
            "label":"ชั้นที่ 4 — รองพื้นทางวัสดุมวลรวม CBR 25% (Aggregate Subbase)",
            "css":"layer-subbase", "emoji":"🔵",
            "mat_opts":["วัสดุมวลรวม CBR>=25% (a=0.10)","Natural Gravel CBR>=20% (a=0.09)",
                        "Sandy Gravel CBR>=15% (a=0.08)","กำหนดเอง"],
            "mat_a":[0.10, 0.09, 0.08, 0.10],
            "D_min_cm":10, "D_max_cm":60,
            "a_def":0.10, "a_min":0.04, "a_max":0.25,
            "m_def":1.1,  "use_def":True,
        },
        {
            "no":5,
            "label":"ชั้นที่ 5 — วัสดุคัดเลือก (Selected Material)",
            "css":"layer-select", "emoji":"🟣",
            "mat_opts":["วัสดุคัดเลือก (a=0.08)","Selected Soil CBR>=10% (a=0.08)",
                        "Lime-stabilized (a=0.10)","Cement-stabilized (a=0.12)","กำหนดเอง"],
            "mat_a":[0.08, 0.08, 0.10, 0.12, 0.08],
            "D_min_cm":0, "D_max_cm":60,
            "a_def":0.08, "a_min":0.04, "a_max":0.20,
            "m_def":1.1,  "use_def":True,
        },
    ]

    col_left, col_right = st.columns([1.35, 0.65])
    layers_config = []

    with col_left:
        for ld in LAYER_DEFS:
            n = ld["no"]
            with st.expander(f"{ld['emoji']} {ld['label']}", expanded=True):
                ca, cb, cc, cd = st.columns([1.5, 0.55, 0.55, 0.45])
                with ca:
                    mat = st.selectbox("วัสดุ", ld["mat_opts"], key=f"mat_{n}")
                    idx = ld["mat_opts"].index(mat)
                    a_default = float(ld["mat_a"][idx])
                with cb:
                    a_val = st.number_input("aᵢ", ld["a_min"], ld["a_max"],
                                             a_default, 0.01, key=f"a_{n}")
                with cc:
                    m_val = st.number_input("mᵢ", 0.40, 1.40,
                                             ld["m_def"], 0.05, key=f"m_{n}",
                                             help="Drainage Coefficient (1.1 = ดีเยี่ยม-ดี)")
                with cd:
                    use_layer = st.checkbox("ใช้", value=ld["use_def"], key=f"use_{n}")

                ce, cf = st.columns(2)
                with ce:
                    D_min_cm = st.number_input("ความหนาขั้นต่ำ (cm)", 0, ld["D_max_cm"],
                                                ld["D_min_cm"] if use_layer else 0,
                                                1, key=f"Dmin_{n}")
                with cf:
                    D_max_cm = st.number_input("ความหนาสูงสุด (cm)", D_min_cm, 150,
                                                ld["D_max_cm"], 1, key=f"Dmax_{n}")

                st.caption(f"SN contribution = aᵢ × mᵢ × Dᵢ = {a_val:.2f} × {m_val:.2f} × Dᵢ")

                layers_config.append({
                    "no": n, "label": ld["label"], "css": ld["css"], "emoji": ld["emoji"],
                    "mat": mat, "a": a_val, "m": m_val,
                    "D_min_in": D_min_cm / 2.54,
                    "D_max_in": D_max_cm / 2.54,
                    "use": use_layer,
                })

    # ── Solve Thicknesses ─────────────────────────────────────────────────────
    SN_cumul = 0.0
    results = []
    for lc in layers_config:
        if not lc["use"]:
            results.append({**lc, "D_in":0.0, "D_cm":0.0, "SN_contrib":0.0})
            continue
        a, m = lc["a"], lc["m"]
        SN_need = max(0.0, SN_req - SN_cumul)
        D_in = lc["D_min_in"]
        if a * m > 0:
            D_needed = SN_need / (a * m)
            D_in = max(lc["D_min_in"], D_needed)
        if lc["D_max_in"] > 0:
            D_in = min(D_in, lc["D_max_in"])
        D_in = math.ceil(D_in * 4) / 4   # round up to nearest 0.25"
        SN_contrib = a * m * D_in
        SN_cumul  += SN_contrib
        results.append({**lc, "D_in":D_in, "D_cm":D_in*2.54, "SN_contrib":SN_contrib})

    SN_provided = sum(r["SN_contrib"] for r in results)
    total_cm    = sum(r["D_cm"] for r in results)

    # ── Cross-Section Diagram ─────────────────────────────────────────────────
    with col_right:
        ok_color = "#4caf50" if SN_provided >= SN_req else "#f44336"
        ok_label = "✅ ผ่านการออกแบบ" if SN_provided >= SN_req else "❌ SN ไม่เพียงพอ"

        html = '<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:10px;padding:16px;font-family:\'IBM Plex Mono\',monospace;">'
        html += '<div style="color:#7ab3d4;font-size:0.7rem;text-align:center;margin-bottom:10px;letter-spacing:1.5px;">PAVEMENT CROSS-SECTION</div>'

        for r in results:
            if not r["use"] or r["D_cm"] == 0:
                continue
            mat_short = r["mat"].split("(")[0].strip()
            html += f'<div class="layer-box {r["css"]}"><span>{r["emoji"]} {mat_short}</span><span><b>{r["D_cm"]:.1f} cm</b> ({r["D_in"]:.2f}")</span></div>'
            html += f'<div style="text-align:center;color:#333;font-size:0.66rem;margin:-2px 0 2px 0;">SN={r["SN_contrib"]:.3f} (a={r["a"]:.2f} m={r["m"]:.2f})</div>'

        html += f'<div class="layer-box layer-subgrade"><span>🟤 Subgrade</span><span>MR={MR:.0f} psi</span></div>'
        html += f'<hr style="border-color:#1e3a5f;margin:10px 0;">'
        html += f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#7ab3d4;"><span>รวม <b style="color:#c8f5c8">{total_cm:.1f} cm</b></span><span>SN = <b style="color:{ok_color}">{SN_provided:.3f}</b></span></div>'
        html += f'<div style="text-align:center;margin-top:8px;font-size:0.9rem;color:{ok_color};font-weight:700;">{ok_label}</div>'
        html += f'<div style="text-align:center;color:#4a7a9b;font-size:0.72rem;">SN required = {SN_req:.3f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        # Bar chart
        active = [r for r in results if r["use"] and r["D_cm"] > 0]
        if active:
            color_map = {
                "layer-wearing":"#c8a020","layer-binder":"#8a7020",
                "layer-base":"#4a7a4a","layer-subbase":"#4a4a8a","layer-select":"#7a4a6a",
            }
            fig_bar = go.Figure(go.Bar(
                x=[r["D_cm"] for r in active],
                y=[r["mat"].split("(")[0].strip()[:22] for r in active],
                orientation='h',
                marker_color=[color_map.get(r["css"],"#555") for r in active],
                text=[f"{r['D_cm']:.1f} cm" for r in active],
                textposition='outside',
            ))
            fig_bar.update_layout(
                title="ความหนาแต่ละชั้น",
                xaxis_title="ความหนา (cm)",
                template="plotly_dark",
                paper_bgcolor="#0a0f1a", plot_bgcolor="#0a0f1a",
                font=dict(family="Sarabun", color="#7ab3d4", size=10),
                height=260, margin=dict(l=10,r=50,t=40,b=10), showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Summary Table ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 สรุปผลการออกแบบ")

    rows = []
    for r in results:
        rows.append({
            "ชั้นที่": r["no"],
            "ชั้นทาง": r["label"].split("—")[1].strip() if "—" in r["label"] else r["label"],
            "วัสดุ": r["mat"].split("(")[0].strip(),
            "aᵢ": f"{r['a']:.3f}",
            "mᵢ": f"{r['m']:.3f}",
            "ความหนา (นิ้ว)": f"{r['D_in']:.2f}" if r["use"] else "—",
            "ความหนา (cm)": f"{r['D_cm']:.1f}" if r["use"] else "—",
            "SN ที่ได้": f"{r['SN_contrib']:.3f}" if r["use"] else "—",
            "ใช้": "✅" if r["use"] else "❌",
        })
    rows.append({
        "ชั้นที่":"","ชั้นทาง":"รวมทั้งหมด","วัสดุ":"","aᵢ":"","mᵢ":"",
        "ความหนา (นิ้ว)": f"{sum(r['D_in'] for r in results):.2f}",
        "ความหนา (cm)": f"{total_cm:.1f}",
        "SN ที่ได้": f"{SN_provided:.3f}", "ใช้":"",
    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if SN_provided >= SN_req:
        st.success(f"✅ โครงสร้างทางผ่าน — SN ที่ได้ {SN_provided:.3f} >= SN ที่ต้องการ {SN_req:.3f}  (เกิน {SN_provided-SN_req:.3f})")
    else:
        st.error(f"❌ SN ไม่เพียงพอ — ขาดอีก {SN_req-SN_provided:.3f}  กรุณาเพิ่มความหนาหรือเปลี่ยนวัสดุ")

    # Donut chart
    st.markdown("### 🥧 สัดส่วน SN แต่ละชั้น")
    active2 = [r for r in results if r["use"] and r["SN_contrib"] > 0]
    if active2:
        c_list = ["#c8a020","#8a7020","#4a7a4a","#4a4a8a","#7a4a6a"]
        fig_pie = go.Figure(go.Pie(
            labels=[r["mat"].split("(")[0].strip()[:25] for r in active2],
            values=[r["SN_contrib"] for r in active2],
            hole=0.55,
            marker_colors=c_list[:len(active2)],
            texttemplate="%{label}<br><b>SN=%{value:.3f}</b>",
        ))
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117",
            font=dict(family="Sarabun", color="#7ab3d4"),
            height=320, showlegend=False,
            annotations=[dict(text=f"SN<br>{SN_provided:.3f}", x=0.5, y=0.5,
                              font_size=16, showarrow=False, font_color="#c8f5c8")],
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Sensitivity Analysis")
    c1, c2 = st.columns(2)

    with c1:
        W18r = np.logspace(4, 9, 80)
        SNr  = [solve_SN(w, ZR, S0, delta_psi, MR) or np.nan for w in W18r]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=W18r, y=SNr, mode='lines', line=dict(color='#0d74e7',width=2.5)))
        fig1.add_vline(x=W18, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"W18={W18:.1e}", annotation_font_color="#f0a030")
        fig1.add_hline(y=SN_req, line_dash="dash", line_color="#4caf50",
                       annotation_text=f"SN={SN_req:.3f}", annotation_font_color="#4caf50")
        fig1.update_layout(title="SN vs W18 (ESAL)", xaxis_title="W18", yaxis_title="SN Required",
                            xaxis_type="log", template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                            font=dict(family="Sarabun",color="#7ab3d4"), height=340)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        MRr = np.linspace(2000, 25000, 80)
        SNm = [solve_SN(W18, ZR, S0, delta_psi, mr) or np.nan for mr in MRr]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=MRr, y=SNm, mode='lines', line=dict(color='#e74c3c',width=2.5)))
        fig2.add_vline(x=MR, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"MR={MR:.0f}", annotation_font_color="#f0a030")
        fig2.update_layout(title="SN vs MR Subgrade", xaxis_title="MR (psi)", yaxis_title="SN Required",
                            template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                            font=dict(family="Sarabun",color="#7ab3d4"), height=340)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        Rr   = [50,60,70,75,80,85,90,91,92,93,94,95,96,97,98,99]
        SNRr = [solve_SN(W18, get_ZR(r), S0, delta_psi, MR) or np.nan for r in Rr]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=Rr, y=SNRr, mode='lines+markers',
                                   line=dict(color='#9b59b6',width=2.5), marker=dict(size=5)))
        fig3.add_vline(x=reliability, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"R={reliability}%", annotation_font_color="#f0a030")
        fig3.update_layout(title="SN vs Reliability", xaxis_title="R (%)", yaxis_title="SN Required",
                            template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                            font=dict(family="Sarabun",color="#7ab3d4"), height=340)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        DPr  = np.linspace(1.0, 3.5, 60)
        SNDr = [solve_SN(W18, ZR, S0, dp, MR) or np.nan for dp in DPr]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=DPr, y=SNDr, mode='lines', line=dict(color='#1abc9c',width=2.5)))
        fig4.add_vline(x=delta_psi, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"DPSI={delta_psi:.1f}", annotation_font_color="#f0a030")
        fig4.update_layout(title="SN vs DPSI", xaxis_title="DPSI", yaxis_title="SN Required",
                            template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                            font=dict(family="Sarabun",color="#7ab3d4"), height=340)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Tables
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 ตารางค่าสัมประสิทธิ์อ้างอิง")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### ชั้น 1: ผิวทางลาดยาง (AC)")
        st.dataframe(pd.DataFrame({
            "วัสดุ":["ผิวทางลาดยาง AC (ค่ากรมทางหลวง)","Dense Graded AC","Polymer Modified AC"],
            "E (MPa)":[2500,3000,4000],
            "a₁":[0.40,0.44,0.46],
            "m₁ (ค่าแนะนำ)":[1.1,1.1,1.1],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### ชั้น 2: พื้นทางซีเมนต์ (CTBAC)")
        st.dataframe(pd.DataFrame({
            "วัสดุ":["พื้นทางซีเมนต์ CTBAC","Cement-treated Base","Lean Concrete"],
            "a₂":[0.18,0.20,0.23],
            "m₂ (ค่าแนะนำ)":[1.1,1.1,1.1],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### ชั้น 3: พื้นทางหินคลุก 80% AC")
        st.dataframe(pd.DataFrame({
            "วัสดุ":["หินคลุก 80% AC","Crushed Stone Base","Dense Graded Gravel"],
            "a₃":[0.13,0.14,0.12],
            "m₃ (ค่าแนะนำ)":[1.1,1.1,1.1],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### ชั้น 4: รองพื้นทางวัสดุมวลรวม CBR 25%")
        st.dataframe(pd.DataFrame({
            "วัสดุ":["วัสดุมวลรวม CBR>=25%","Natural Gravel CBR>=20%","Sandy Gravel CBR>=15%"],
            "a₄":[0.10,0.09,0.08],
            "m₄ (ค่าแนะนำ)":[1.1,1.1,1.1],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### ชั้น 5: วัสดุคัดเลือก (Selected Material)")
        st.dataframe(pd.DataFrame({
            "วัสดุ":["วัสดุคัดเลือก","Selected Soil CBR>=10%","Lime-stabilized","Cement-stabilized"],
            "a₅":[0.08,0.08,0.10,0.12],
            "m₅ (ค่าแนะนำ)":[1.1,1.1,1.1,1.1],
        }), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Drainage Coefficient (m)")
        dr = []
        for q,sd in DRAIN_COEFF.items():
            for s,v in sd.items():
                dr.append({"คุณภาพระบาย":q,"% ชื้น":s,"m":v})
        st.dataframe(pd.DataFrame(dr), use_container_width=True, hide_index=True)

        st.markdown("#### ZR — Standard Normal Deviate")
        st.dataframe(pd.DataFrame({
            "R (%)":[50,60,70,75,80,85,90,95,98,99],
            "ZR":[ZR_TABLE[r] for r in [50,60,70,75,80,85,90,95,98,99]],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### MR จาก CBR")
        cbr_list = [2,3,4,5,6,8,10,12,15,20,25]
        st.dataframe(pd.DataFrame({
            "CBR (%)":cbr_list,
            "MR (psi)":[cbr_to_MR(c) for c in cbr_list],
            "MR (MPa)":[round(cbr_to_MR(c)/145.038,1) for c in cbr_list],
        }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Theory
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ ทฤษฎีและสูตร AASHTO 1993 — ระบบ 5 ชั้น")

    st.markdown("""
    <div class="formula-box">
    <b style="color:#a8d0f0;">AASHTO 1993 Design Equation:</b><br><br>
    log(W18) = ZR x S0 + 9.36 x log(SN+1) - 0.20<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ log[DPSI/(4.2-1.5)] / [0.40 + 1094/(SN+1)^5.19]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 2.32 x log(MR) - 8.07<br><br>
    <b style="color:#a8d0f0;">Structural Number — 5 ชั้น (กรมทางหลวง):</b><br>
    SN = a1*m1*D1 + a2*m2*D2 + a3*m3*D3 + a4*m4*D4 + a5*m5*D5<br><br>
    &nbsp;&nbsp; D1 = ผิวทางลาดยาง AC          a1=0.40, m1=1.1<br>
    &nbsp;&nbsp; D2 = พื้นทางซีเมนต์ CTBAC      a2=0.18, m2=1.1<br>
    &nbsp;&nbsp; D3 = พื้นทางหินคลุก 80% AC      a3=0.13, m3=1.1<br>
    &nbsp;&nbsp; D4 = รองพื้นทางวัสดุมวลรวม CBR25% a4=0.10, m4=1.1<br>
    &nbsp;&nbsp; D5 = วัสดุคัดเลือก               a5=0.08, m5=1.1<br><br>
    <b style="color:#a8d0f0;">MR จาก CBR:</b>  MR (psi) = 1,500 x CBR (%)
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box">
        <b style="color:#a8d0f0;">โครงสร้าง 5 ชั้นทาง (กรมทางหลวง):</b><br>
        🟡 <b>ชั้นที่ 1 ผิวทางลาดยาง (AC)</b> — รับแรงเสียดทาน a=0.40 m=1.1<br>
        🟠 <b>ชั้นที่ 2 พื้นทางซีเมนต์ (CTBAC)</b> — Cement Treated Base a=0.18 m=1.1<br>
        🟢 <b>ชั้นที่ 3 พื้นทางหินคลุก 80% AC</b> — กระจายน้ำหนัก a=0.13 m=1.1<br>
        🔵 <b>ชั้นที่ 4 รองพื้นทางวัสดุมวลรวม CBR25%</b> — a=0.10 m=1.1<br>
        🟣 <b>ชั้นที่ 5 วัสดุคัดเลือก</b> — Selected Material a=0.08 m=1.1<br>
        🟤 <b>ดินพื้นทาง (Subgrade)</b> — ไม่นับใน SN
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <b style="color:#a8d0f0;">ขั้นตอนออกแบบ 5 ชั้น:</b><br>
        1. กำหนด W18 จากการสำรวจจราจร<br>
        2. เลือก Reliability (R) ตามประเภทถนน<br>
        3. กำหนด S0, pi, pt<br>
        4. วัด MR ดินพื้นทาง (หรือแปลงจาก CBR)<br>
        5. แก้สมการหา SN required<br>
        6. เลือกวัสดุแต่ละชั้น กำหนด aᵢ, mᵢ<br>
        7. คำนวณ Dᵢ ให้ SN สะสม >= SN required<br>
        8. ตรวจสอบความหนาขั้นต่ำ/สูงสุด<br>
        9. ตรวจสอบ SN provided >= SN required
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:12px">
    <b style="color:#a8d0f0;">อ้างอิง:</b><br>
    AASHTO (1993). Guide for Design of Pavement Structures. Washington D.C.<br>
    กรมทางหลวง (2566). มาตรฐานการออกแบบโครงสร้างชั้นทาง. กระทรวงคมนาคม.<br>
    Huang, Y.H. (2004). Pavement Analysis and Design, 2nd Ed. Pearson.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e3a5f;margin:32px 0 12px 0;">
<div style="text-align:center;color:#2d5a8a;font-size:0.78rem;font-family:'Sarabun',sans-serif;">
    🛣️ กรมทางหลวง — Department of Highways Thailand |
    AASHTO 1993 Flexible Pavement Design (5-Layer System)<br>
    <span style="font-size:0.68rem;color:#1e3a5f;">
        สร้างด้วย Streamlit + Python | สำหรับการศึกษาและประกอบการออกแบบเบื้องต้นเท่านั้น
    </span>
</div>
""", unsafe_allow_html=True)
