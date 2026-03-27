"""
AASHTO 1993 Flexible Pavement Design
ออกแบบผิวทางลาดยาง ตามวิธี AASHTO 1993
กรมทางหลวง - Department of Highways Thailand
"""

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.optimize import brentq

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AASHTO 1993 | ออกแบบผิวทาง",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif;
}

.main { background-color: #0d1117; }

h1, h2, h3 { font-family: 'Sarabun', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 60%, #0a1628 100%);
    border: 1px solid #2d5a8a;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(13,97,166,0.3);
}
.header-box h1 {
    color: #e6f3ff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 4px 0;
}
.header-box p {
    color: #7ab3d4;
    margin: 0;
    font-size: 0.95rem;
}

/* Result Cards */
.result-card {
    background: linear-gradient(135deg, #1a2e1a 0%, #0f1f0f 100%);
    border: 1px solid #2d6b2d;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(45,107,45,0.2);
}
.result-card .label {
    color: #7dd87d;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.result-card .value {
    color: #c8f5c8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
}
.result-card .unit {
    color: #7dd87d;
    font-size: 0.85rem;
}

.result-card-blue {
    background: linear-gradient(135deg, #1a2a3e 0%, #0f1a2d 100%);
    border: 1px solid #2d5a8a;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.result-card-blue .label { color: #7ab3d4; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.result-card-blue .value { color: #b3d9f5; font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 700; }
.result-card-blue .unit { color: #7ab3d4; font-size: 0.85rem; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #1e3a5f 0%, transparent 100%);
    border-left: 4px solid #0d74e7;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0 12px 0;
}
.section-header h3 {
    color: #a8d0f0;
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
}

/* Layer diagram */
.layer-box {
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.layer-ac { background: #1a1505; border: 1px solid #8a7020; color: #d4b84a; }
.layer-base { background: #0d1a0d; border: 1px solid #4a7a4a; color: #8ac88a; }
.layer-subbase { background: #0f0f1a; border: 1px solid #4a4a8a; color: #8a8ac8; }
.layer-subgrade { background: #1a0f0a; border: 1px solid #8a5a3a; color: #c8a07a; }

/* Info box */
.info-box {
    background: #111d2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    font-size: 0.85rem;
    color: #8aafcc;
    line-height: 1.6;
}
.formula-box {
    background: #0a0f1a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #7ab3d4;
    line-height: 2;
}

/* Stmetric override */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #c8f5c8 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e3a5f;
}

/* Tabs */
[data-baseweb="tab"] {
    font-family: 'Sarabun', sans-serif !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #111d2e;
    border-radius: 8px;
    padding: 4px;
}

/* Warning/Success */
.design-ok { color: #4caf50; font-weight: 700; }
.design-fail { color: #f44336; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── AASHTO 1993 Core Functions ───────────────────────────────────────────────

# ZR values for different reliability levels
ZR_TABLE = {
    50: -0.000,
    60: -0.253,
    70: -0.524,
    75: -0.674,
    80: -0.842,
    85: -1.037,
    90: -1.282,
    91: -1.341,
    92: -1.405,
    93: -1.476,
    94: -1.555,
    95: -1.645,
    96: -1.751,
    97: -1.881,
    98: -2.054,
    99: -2.327,
    99.9: -3.090,
}

def get_ZR(reliability):
    """Interpolate ZR from reliability"""
    levels = sorted(ZR_TABLE.keys())
    if reliability <= levels[0]:
        return ZR_TABLE[levels[0]]
    if reliability >= levels[-1]:
        return ZR_TABLE[levels[-1]]
    for i in range(len(levels)-1):
        if levels[i] <= reliability <= levels[i+1]:
            r1, r2 = levels[i], levels[i+1]
            z1, z2 = ZR_TABLE[r1], ZR_TABLE[r2]
            return z1 + (z2 - z1) * (reliability - r1) / (r2 - r1)

def aashto_lhs(SN, W18, ZR, S0, delta_psi, MR):
    """
    AASHTO 1993 Equation — compute left-hand side minus right-hand side = 0
    log(W18) = ZR*S0 + 9.36*log(SN+1) - 0.20 + f(ΔPSI,SN) + 2.32*log(MR) - 8.07
    """
    log_W18 = math.log10(W18)
    
    term1 = ZR * S0
    term2 = 9.36 * math.log10(SN + 1) - 0.20
    
    # PSI loss term
    psi_ratio = delta_psi / (4.2 - 1.5)
    if psi_ratio <= 0:
        return float('inf')
    term3_num = math.log10(psi_ratio)
    term3_den = 0.40 + 1094 / ((SN + 1) ** 5.19)
    term3 = term3_num / term3_den
    
    term4 = 2.32 * math.log10(MR) - 8.07
    
    rhs = term1 + term2 + term3 + term4
    return rhs - log_W18

def solve_SN(W18, ZR, S0, delta_psi, MR):
    """Solve for SN using Brent's method"""
    try:
        SN = brentq(aashto_lhs, 0.01, 30.0, 
                    args=(W18, ZR, S0, delta_psi, MR),
                    xtol=1e-6, maxiter=200)
        return SN
    except Exception as e:
        return None

def calc_W18(SN, ZR, S0, delta_psi, MR):
    """Calculate W18 from SN"""
    term1 = ZR * S0
    term2 = 9.36 * math.log10(SN + 1) - 0.20
    psi_ratio = delta_psi / (4.2 - 1.5)
    term3_num = math.log10(psi_ratio)
    term3_den = 0.40 + 1094 / ((SN + 1) ** 5.19)
    term3 = term3_num / term3_den
    term4 = 2.32 * math.log10(MR) - 8.07
    log_W18 = term1 + term2 + term3 + term4
    return 10 ** log_W18

def subgrade_MR_from_CBR(CBR):
    """MR (psi) from CBR — AASHTO approximation"""
    return 1500 * CBR

def layer_coefficient_AC(E_AC_psi=None, E_AC_MPa=None, method="modulus"):
    """
    Layer coefficient a1 for asphalt concrete
    Using regression: a1 = 0.001977*(E^0.5) where E in psi (approx)
    Common values: E=450,000 psi → a1≈0.44
    """
    if E_AC_MPa is not None:
        E_AC_psi = E_AC_MPa * 145.038
    if E_AC_psi is None:
        E_AC_psi = 450000
    # AASHTO chart approximation
    a1 = 0.001977 * (E_AC_psi ** 0.5) / 100
    # Simplified regression from AASHTO 1993 charts
    # More accurate: a1 = 0.44 at 450,000 psi
    if E_AC_psi >= 400000:
        a1 = 0.40 + (E_AC_psi - 400000) / (450000 - 400000) * 0.04
    else:
        a1 = 0.30 + (E_AC_psi - 200000) / (400000 - 200000) * 0.10
    return round(max(0.20, min(0.50, a1)), 3)

# ─── Drainage Coefficient Table ───────────────────────────────────────────────
DRAIN_COEFF = {
    "ดีเยี่ยม (< 2 ชม.)": {"< 1%": 1.40, "1-5%": 1.35, "5-25%": 1.30, "> 25%": 1.20},
    "ดี (2-24 ชม.)":       {"< 1%": 1.35, "1-5%": 1.25, "5-25%": 1.15, "> 25%": 1.00},
    "ปานกลาง (1-7 วัน)":  {"< 1%": 1.25, "1-5%": 1.15, "5-25%": 1.05, "> 25%": 0.80},
    "แย่ (1-4 สัปดาห์)":  {"< 1%": 1.15, "1-5%": 1.05, "5-25%": 0.80, "> 25%": 0.60},
    "แย่มาก (> 1 เดือน)": {"< 1%": 1.05, "1-5%": 0.95, "5-25%": 0.75, "> 25%": 0.40},
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <div style="font-size:2.5rem;">🛣️</div>
        <div style="color:#7ab3d4; font-weight:700; font-size:1rem; letter-spacing:1px;">AASHTO 1993</div>
        <div style="color:#4a7a9b; font-size:0.75rem;">Flexible Pavement Design</div>
    </div>
    <hr style="border-color:#1e3a5f; margin:8px 0;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ พารามิเตอร์การออกแบบ")

    # ── Traffic ──
    st.markdown('<div class="section-header"><h3>🚛 ปริมาณจราจร</h3></div>', unsafe_allow_html=True)
    
    W18_input_method = st.radio("วิธีกรอก ESAL", ["กรอก W18 โดยตรง", "คำนวณจากปริมาณจราจร"], horizontal=True)
    
    if W18_input_method == "กรอก W18 โดยตรง":
        W18 = st.number_input("W18 (ESAL ตลอดอายุออกแบบ)", 
                               min_value=1e4, max_value=1e9, value=5e6,
                               format="%.2e", step=1e5)
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            AADT = st.number_input("AADT (คัน/วัน)", min_value=100, max_value=200000, value=5000, step=100)
            T_pct = st.number_input("% รถบรรทุก", min_value=1.0, max_value=60.0, value=15.0, step=1.0)
        with col_t2:
            design_life = st.number_input("อายุออกแบบ (ปี)", min_value=5, max_value=40, value=20, step=5)
            TF = st.number_input("Truck Factor (LEF)", min_value=0.1, max_value=5.0, value=0.5, step=0.05)
        growth_rate = st.number_input("อัตราการเติบโต (%/ปี)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
        
        # Growth factor
        if growth_rate > 0:
            GF = ((1 + growth_rate/100)**design_life - 1) / (growth_rate/100)
        else:
            GF = design_life
        
        trucks_per_day = AADT * (T_pct / 100)
        W18 = trucks_per_day * 365 * GF * TF
        st.info(f"**W18 = {W18:,.0f} ESAL**")

    # ── Reliability ──
    st.markdown('<div class="section-header"><h3>📊 ความน่าเชื่อถือ</h3></div>', unsafe_allow_html=True)
    
    highway_class = st.selectbox("ประเภททาง", 
        ["ทางหลวงระหว่างเมือง (R=95%)", "ทางหลวงสายหลัก (R=90%)", 
         "ทางหลวงสายรอง (R=85%)", "ถนนในเมืองสายหลัก (R=95%)",
         "ถนนในเมืองสายรอง (R=80%)", "กำหนดเอง"])
    
    if highway_class == "กำหนดเอง":
        reliability = st.slider("Reliability (%)", 50, 99, 95)
    else:
        r_map = {
            "ทางหลวงระหว่างเมือง (R=95%)": 95,
            "ทางหลวงสายหลัก (R=90%)": 90,
            "ทางหลวงสายรอง (R=85%)": 85,
            "ถนนในเมืองสายหลัก (R=95%)": 95,
            "ถนนในเมืองสายรอง (R=80%)": 80,
            "กำหนดเอง": 90,
        }
        reliability = r_map[highway_class]
        st.info(f"R = {reliability}%")
    
    ZR = get_ZR(reliability)
    S0 = st.number_input("S₀ (Overall Std. Deviation)", min_value=0.30, max_value=0.60, value=0.45, step=0.01,
                          help="ค่าแนะนำ: 0.40–0.50 สำหรับผิวทางลาดยาง")

    # ── Serviceability ──
    st.markdown('<div class="section-header"><h3>📉 Serviceability</h3></div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pi = st.number_input("p_i (เริ่มต้น)", min_value=3.5, max_value=5.0, value=4.2, step=0.1)
    with col_p2:
        pt = st.number_input("p_t (สิ้นสุด)", min_value=1.5, max_value=3.5, value=2.5, step=0.1)
    delta_psi = pi - pt
    st.info(f"ΔPSI = {delta_psi:.1f}")

    # ── Subgrade ──
    st.markdown('<div class="section-header"><h3>🏔️ ดินพื้นทาง (Subgrade)</h3></div>', unsafe_allow_html=True)
    
    MR_method = st.radio("กรอกค่า MR จาก", ["CBR (%)", "MR โดยตรง (psi)", "MR โดยตรง (MPa)"], horizontal=False)
    
    if MR_method == "CBR (%)":
        CBR = st.number_input("CBR ดินพื้นทาง (%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
        MR = subgrade_MR_from_CBR(CBR)
        st.info(f"MR = {MR:,.0f} psi ({MR/145.038:.1f} MPa)")
    elif MR_method == "MR โดยตรง (psi)":
        MR = st.number_input("MR (psi)", min_value=1000, max_value=30000, value=7500, step=500)
    else:
        MR_mpa = st.number_input("MR (MPa)", min_value=10, max_value=200, value=52, step=5)
        MR = MR_mpa * 145.038


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛣️ ออกแบบผิวทางลาดยาง — AASHTO 1993</h1>
    <p>Department of Highways Thailand &nbsp;|&nbsp; กรมทางหลวง &nbsp;|&nbsp; Flexible Pavement Design</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📐 ออกแบบโครงสร้างทาง", "📊 กราฟ Sensitivity", "📋 ตาราง Layer Coefficient", "ℹ️ ทฤษฎีและสูตร"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Design
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Solve required SN
    SN_req = solve_SN(W18, ZR, S0, delta_psi, MR)

    # Top result row
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown(f"""<div class="result-card">
            <div class="label">Structural Number</div>
            <div class="value">{SN_req:.3f}</div>
            <div class="unit">SN required</div>
        </div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""<div class="result-card-blue">
            <div class="label">Design ESAL (W18)</div>
            <div class="value">{W18:,.0f}</div>
            <div class="unit">18-kip ESAL</div>
        </div>""", unsafe_allow_html=True)
    with col_r3:
        st.markdown(f"""<div class="result-card-blue">
            <div class="label">Reliability (ZR)</div>
            <div class="value">{reliability}%</div>
            <div class="unit">ZR = {ZR:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col_r4:
        st.markdown(f"""<div class="result-card-blue">
            <div class="label">Subgrade MR</div>
            <div class="value">{MR:,.0f}</div>
            <div class="unit">psi = {MR/145.038:.1f} MPa</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Layer Design ──────────────────────────────────────────────────────────
    st.markdown("### 🏗️ ออกแบบความหนาชั้นทาง")

    col_lay, col_dia = st.columns([1.2, 0.8])

    with col_lay:
        st.markdown("**ชั้นที่ 1 — Asphalt Concrete (ผิวทางลาดยาง)**")
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            E_AC = st.number_input("E_AC (MPa)", 1000, 8000, 3000, 100,
                                    help="โมดูลัสของ AC ที่อุณหภูมิออกแบบ")
        with col1b:
            a1 = st.number_input("a₁ (Layer Coeff.)", 0.20, 0.50, 0.44, 0.01,
                                  help="ค่าสัมประสิทธิ์ชั้นทาง AC")
        with col1c:
            D1_min = st.number_input("D₁ min (cm)", 5, 30, 10, 1,
                                      help="ความหนาขั้นต่ำของผิวทาง AC")

        st.markdown("**ชั้นที่ 2 — Base Course (ชั้นรองผิวทาง)**")
        col2a, col2b, col2c, col2d = st.columns(4)
        with col2a:
            base_type = st.selectbox("วัสดุ Base", 
                ["Crushed Stone (a₂=0.14)", "Dense Graded (a₂=0.12)", 
                 "Lime-treated (a₂=0.18)", "Asphalt Treated (a₂=0.30)", "กำหนดเอง"])
        with col2b:
            a2_default = {"Crushed Stone (a₂=0.14)": 0.14, "Dense Graded (a₂=0.12)": 0.12,
                          "Lime-treated (a₂=0.18)": 0.18, "Asphalt Treated (a₂=0.30)": 0.30, "กำหนดเอง": 0.14}
            a2 = st.number_input("a₂", 0.05, 0.40, float(a2_default[base_type]), 0.01)
        with col2c:
            drain_qual2 = st.selectbox("Drainage Base", list(DRAIN_COEFF.keys()), index=1)
            sat_pct2 = st.selectbox("% Saturation Base", ["< 1%", "1-5%", "5-25%", "> 25%"], index=1)
            m2 = DRAIN_COEFF[drain_qual2][sat_pct2]
        with col2d:
            D2_min = st.number_input("D₂ min (cm)", 10, 40, 20, 2)

        st.markdown("**ชั้นที่ 3 — Subbase Course (ชั้นพื้นทาง)**")
        col3a, col3b, col3c, col3d = st.columns(4)
        with col3a:
            sub_type = st.selectbox("วัสดุ Subbase", 
                ["Natural Gravel (a₃=0.11)", "Sandy Gravel (a₃=0.10)",
                 "Lime-treated (a₃=0.13)", "กำหนดเอง"])
        with col3b:
            a3_default = {"Natural Gravel (a₃=0.11)": 0.11, "Sandy Gravel (a₃=0.10)": 0.10,
                          "Lime-treated (a₃=0.13)": 0.13, "กำหนดเอง": 0.11}
            a3 = st.number_input("a₃", 0.05, 0.30, float(a3_default[sub_type]), 0.01)
        with col3c:
            drain_qual3 = st.selectbox("Drainage Subbase", list(DRAIN_COEFF.keys()), index=1)
            sat_pct3 = st.selectbox("% Saturation Sub", ["< 1%", "1-5%", "5-25%", "> 25%"], index=1)
            m3 = DRAIN_COEFF[drain_qual3][sat_pct3]
        with col3d:
            D3_min = st.number_input("D₃ min (cm)", 0, 40, 15, 2)

    # ── Solve for layer thicknesses ───────────────────────────────────────────
    # Convert to inches for AASHTO
    D1_min_in = D1_min / 2.54
    D2_min_in = D2_min / 2.54
    D3_min_in = D3_min / 2.54

    # Iterative layer design
    # SN1 = a1*D1 → D1
    # SN2 = a1*D1 + a2*m2*D2 → D2
    # SN3 = a1*D1 + a2*m2*D2 + a3*m3*D3 → D3

    # SN needed at subgrade level = SN_req
    # SN needed at base level: solve with MR_base (approx)
    # Simplified: use direct layer approach

    # Layer 1: D1
    D1_calc_in = max(D1_min_in, SN_req / a1 * 0.33)  # rough start
    SN1 = a1 * D1_calc_in
    
    # Actually use proper method: compute SN for each layer interface
    # Use simplified subgrade/base MR estimation
    MR_base = min(30000, MR * 2.5)   # approximate
    MR_subbase = min(20000, MR * 1.8) # approximate

    SN1_req = solve_SN(W18, ZR, S0, delta_psi, MR_subbase) if MR_subbase > 0 else SN_req * 0.6
    SN2_req = solve_SN(W18, ZR, S0, delta_psi, MR_base) if MR_base > 0 else SN_req * 0.4

    if SN1_req is None: SN1_req = SN_req * 0.6
    if SN2_req is None: SN2_req = SN_req * 0.4

    # D1
    D1_in = max(D1_min_in, SN2_req / a1)
    D1_in = math.ceil(D1_in * 2) / 2  # round to 0.5"
    SN1_prov = a1 * D1_in

    # D2
    SN2_need = max(0, SN1_req - SN1_prov)
    if a2 * m2 > 0:
        D2_in = max(D2_min_in, SN2_need / (a2 * m2))
    else:
        D2_in = D2_min_in
    D2_in = math.ceil(D2_in * 2) / 2
    SN2_prov = a2 * m2 * D2_in

    # D3
    SN3_need = max(0, SN_req - SN1_prov - SN2_prov)
    if a3 * m3 > 0:
        D3_in = max(D3_min_in, SN3_need / (a3 * m3))
    else:
        D3_in = D3_min_in
    D3_in = math.ceil(D3_in * 2) / 2
    SN3_prov = a3 * m3 * D3_in

    # Convert back to cm
    D1_cm = D1_in * 2.54
    D2_cm = D2_in * 2.54
    D3_cm = D3_in * 2.54
    total_cm = D1_cm + D2_cm + D3_cm

    SN_provided = SN1_prov + SN2_prov + SN3_prov

    with col_dia:
        # Pavement cross-section diagram
        ok = "✅ ผ่าน" if SN_provided >= SN_req else "❌ ไม่ผ่าน"
        color_ok = "#4caf50" if SN_provided >= SN_req else "#f44336"
        
        st.markdown(f"""
        <div style="background:#0a0f1a; border:1px solid #1e3a5f; border-radius:10px; padding:16px; font-family:'IBM Plex Mono',monospace;">
            <div style="color:#7ab3d4; font-size:0.75rem; text-align:center; margin-bottom:10px; letter-spacing:1px;">PAVEMENT CROSS-SECTION</div>
            
            <div class="layer-box layer-ac">
                <span>🟡 AC Surface</span>
                <span><b>{D1_cm:.1f} cm</b> ({D1_in:.1f}")</span>
            </div>
            <div style="text-align:center; color:#333; font-size:0.7rem;">SN₁ = {SN1_prov:.3f}</div>
            
            <div class="layer-box layer-base">
                <span>🟢 Base Course</span>
                <span><b>{D2_cm:.1f} cm</b> ({D2_in:.1f}")</span>
            </div>
            <div style="text-align:center; color:#333; font-size:0.7rem;">SN₂ = {SN2_prov:.3f}</div>
            
            <div class="layer-box layer-subbase">
                <span>🔵 Subbase</span>
                <span><b>{D3_cm:.1f} cm</b> ({D3_in:.1f}")</span>
            </div>
            <div style="text-align:center; color:#333; font-size:0.7rem;">SN₃ = {SN3_prov:.3f}</div>
            
            <div class="layer-box layer-subgrade">
                <span>🟤 Subgrade (CBR={CBR if MR_method=="CBR (%)" else "—"}%)</span>
                <span>MR={MR:.0f} psi</span>
            </div>
            
            <hr style="border-color:#1e3a5f; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; color:#7ab3d4; font-size:0.8rem;">
                <span>รวมความหนา: <b style="color:#c8f5c8">{total_cm:.1f} cm</b></span>
                <span>SN ที่ได้: <b style="color:{color_ok}">{SN_provided:.3f}</b></span>
            </div>
            <div style="text-align:center; margin-top:8px; font-size:1rem; color:{color_ok}; font-weight:700;">{ok}</div>
            <div style="text-align:center; color:#4a7a9b; font-size:0.75rem;">SN required = {SN_req:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary Table ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 สรุปผลการออกแบบ")

    df_result = pd.DataFrame({
        "ชั้นทาง": ["AC Surface (ผิวทาง)", "Base Course (รองผิวทาง)", "Subbase Course (พื้นทาง)", "**รวมทั้งหมด**"],
        "วัสดุ": ["Asphalt Concrete", base_type.split("(")[0].strip(), sub_type.split("(")[0].strip(), "—"],
        "a (Layer Coeff.)": [f"{a1:.3f}", f"{a2:.3f}", f"{a3:.3f}", "—"],
        "m (Drainage)": ["1.000", f"{m2:.3f}", f"{m3:.3f}", "—"],
        "ความหนา (นิ้ว)": [f"{D1_in:.2f}\"", f"{D2_in:.2f}\"", f"{D3_in:.2f}\"", f"{D1_in+D2_in+D3_in:.2f}\""],
        "ความหนา (cm)": [f"{D1_cm:.1f}", f"{D2_cm:.1f}", f"{D3_cm:.1f}", f"**{total_cm:.1f}**"],
        "SN ที่ได้": [f"{SN1_prov:.3f}", f"{SN2_prov:.3f}", f"{SN3_prov:.3f}", f"**{SN_provided:.3f}**"],
    })

    st.dataframe(df_result, use_container_width=True, hide_index=True)

    # Check status
    if SN_provided >= SN_req:
        st.success(f"✅ โครงสร้างทางผ่านการออกแบบ — SN ที่ได้ ({SN_provided:.3f}) ≥ SN ที่ต้องการ ({SN_req:.3f})")
    else:
        diff = SN_req - SN_provided
        st.error(f"❌ SN ไม่เพียงพอ — ขาด {diff:.3f} กรุณาเพิ่มความหนาชั้นทาง")

    # ── Design Parameters Summary ─────────────────────────────────────────────
    with st.expander("📋 สรุปพารามิเตอร์ที่ใช้ออกแบบ"):
        col_p1, col_p2, col_p3 = st.columns(3)
        params = {
            "W18 (ESAL)": f"{W18:,.0f}",
            "Reliability (R)": f"{reliability}%",
            "ZR": f"{ZR:.4f}",
            "S₀": f"{S0:.2f}",
            "pᵢ (Initial PSI)": f"{pi:.1f}",
            "pₜ (Terminal PSI)": f"{pt:.1f}",
            "ΔPSI": f"{delta_psi:.1f}",
            "MR Subgrade": f"{MR:,.0f} psi",
            "SN Required": f"{SN_req:.4f}",
        }
        items = list(params.items())
        for i, (k, v) in enumerate(items):
            [col_p1, col_p2, col_p3][i % 3].metric(k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Sensitivity Charts
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 กราฟ Sensitivity Analysis")

    col_g1, col_g2 = st.columns(2)

    # Chart 1: SN vs W18
    with col_g1:
        W18_range = np.logspace(4, 9, 80)
        SN_range = []
        for w in W18_range:
            sn = solve_SN(w, ZR, S0, delta_psi, MR)
            SN_range.append(sn if sn else np.nan)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=W18_range, y=SN_range,
            mode='lines', line=dict(color='#0d74e7', width=2.5),
            name='SN Required'
        ))
        fig1.add_vline(x=W18, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"W18={W18:.1e}", annotation_font_color="#f0a030")
        fig1.add_hline(y=SN_req, line_dash="dash", line_color="#4caf50",
                       annotation_text=f"SN={SN_req:.3f}", annotation_font_color="#4caf50")
        fig1.update_layout(
            title="SN Required vs Design ESAL (W18)",
            xaxis_title="W18 (ESAL)", yaxis_title="SN Required",
            xaxis_type="log", template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
            font=dict(family="Sarabun", color="#7ab3d4"),
            height=380,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: SN vs MR
    with col_g2:
        MR_range = np.linspace(2000, 25000, 80)
        SN_MR = []
        for mr in MR_range:
            sn = solve_SN(W18, ZR, S0, delta_psi, mr)
            SN_MR.append(sn if sn else np.nan)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=MR_range, y=SN_MR,
            mode='lines', line=dict(color='#e74c3c', width=2.5),
            name='SN Required'
        ))
        fig2.add_vline(x=MR, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"MR={MR:.0f}", annotation_font_color="#f0a030")
        fig2.update_layout(
            title="SN Required vs Subgrade MR",
            xaxis_title="MR Subgrade (psi)", yaxis_title="SN Required",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
            font=dict(family="Sarabun", color="#7ab3d4"),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_g3, col_g4 = st.columns(2)

    # Chart 3: SN vs Reliability
    with col_g3:
        R_range = [50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        SN_R = []
        for r in R_range:
            zr_temp = get_ZR(r)
            sn = solve_SN(W18, zr_temp, S0, delta_psi, MR)
            SN_R.append(sn if sn else np.nan)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=R_range, y=SN_R,
            mode='lines+markers', line=dict(color='#9b59b6', width=2.5),
            marker=dict(size=6),
            name='SN Required'
        ))
        fig3.add_vline(x=reliability, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"R={reliability}%", annotation_font_color="#f0a030")
        fig3.update_layout(
            title="SN Required vs Reliability (%)",
            xaxis_title="Reliability (%)", yaxis_title="SN Required",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
            font=dict(family="Sarabun", color="#7ab3d4"),
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: SN vs ΔPSI
    with col_g4:
        dpsi_range = np.linspace(1.0, 3.5, 60)
        SN_psi = []
        for dp in dpsi_range:
            sn = solve_SN(W18, ZR, S0, dp, MR)
            SN_psi.append(sn if sn else np.nan)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=dpsi_range, y=SN_psi,
            mode='lines', line=dict(color='#1abc9c', width=2.5),
            name='SN Required'
        ))
        fig4.add_vline(x=delta_psi, line_dash="dash", line_color="#f0a030",
                       annotation_text=f"ΔPSI={delta_psi:.1f}", annotation_font_color="#f0a030")
        fig4.update_layout(
            title="SN Required vs ΔPSI",
            xaxis_title="ΔPSI (pᵢ − pₜ)", yaxis_title="SN Required",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
            font=dict(family="Sarabun", color="#7ab3d4"),
            height=380,
        )
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Layer Coefficient Tables
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 ตารางค่าสัมประสิทธิ์ชั้นทาง (Layer Coefficients)")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### a₁ — Asphalt Concrete (Surface Course)")
        df_a1 = pd.DataFrame({
            "E_AC (MPa)": [1000, 1500, 2000, 3000, 3500, 4000, 5000],
            "E_AC (psi)": ["145,038", "217,557", "290,076", "435,114", "507,633", "580,152", "725,190"],
            "a₁ (approx.)": [0.34, 0.37, 0.39, 0.42, 0.43, 0.44, 0.46],
        })
        st.dataframe(df_a1, use_container_width=True, hide_index=True)

        st.markdown("#### a₂ — Base Course")
        df_a2 = pd.DataFrame({
            "วัสดุ Base": [
                "Dense-graded crushed stone", "Dense-graded crushed gravel",
                "Lime-treated", "Cement-treated", "Bituminous-treated",
                "Recycled asphalt pavement"
            ],
            "CBR / E_base": ["CBR≥100%", "CBR≥80%", "—", "—", "—", "—"],
            "a₂": [0.14, 0.12, 0.18, 0.20, 0.30, 0.12],
        })
        st.dataframe(df_a2, use_container_width=True, hide_index=True)

        st.markdown("#### a₃ — Subbase Course")
        df_a3 = pd.DataFrame({
            "วัสดุ Subbase": [
                "Sandy gravel (CBR≥20%)", "Natural gravel (CBR≥25%)",
                "Lime-treated subbase", "Cement-treated subbase"
            ],
            "a₃": [0.10, 0.11, 0.13, 0.15],
        })
        st.dataframe(df_a3, use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown("#### m — Drainage Coefficients")
        drain_rows = []
        for qual, sat_dict in DRAIN_COEFF.items():
            for sat, val in sat_dict.items():
                drain_rows.append({"คุณภาพระบายน้ำ": qual, "% เวลาชื้น": sat, "m": val})
        df_drain = pd.DataFrame(drain_rows)
        st.dataframe(df_drain, use_container_width=True, hide_index=True)

        st.markdown("#### ZR — Reliability Factor")
        df_zr = pd.DataFrame({
            "Reliability (%)": list(ZR_TABLE.keys()),
            "ZR": list(ZR_TABLE.values()),
        })
        st.dataframe(df_zr, use_container_width=True, hide_index=True)

        st.markdown("#### MR — จาก CBR")
        cbr_vals = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
        df_cbr = pd.DataFrame({
            "CBR (%)": cbr_vals,
            "MR (psi)": [subgrade_MR_from_CBR(c) for c in cbr_vals],
            "MR (MPa)": [round(subgrade_MR_from_CBR(c)/145.038, 1) for c in cbr_vals],
        })
        st.dataframe(df_cbr, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Theory
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ ทฤษฎีและสูตร AASHTO 1993")
    
    st.markdown("""
    <div class="formula-box">
    <b style="color:#a8d0f0;">AASHTO 1993 Design Equation (Flexible Pavement):</b><br><br>
    log(W₁₈) = Z_R × S₀ + 9.36 × log(SN+1) - 0.20<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ log[ΔPSI/(4.2-1.5)] / [0.40 + 1094/(SN+1)⁵·¹⁹]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 2.32 × log(M_R) - 8.07<br><br>
    <b style="color:#a8d0f0;">Structural Number:</b><br>
    SN = a₁D₁ + a₂m₂D₂ + a₃m₃D₃<br><br>
    <b style="color:#a8d0f0;">MR from CBR (AASHTO approximation):</b><br>
    M_R (psi) = 1,500 × CBR (%)
    </div>
    """, unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("""
        <div class="info-box">
        <b style="color:#a8d0f0;">ตัวแปรในสมการ:</b><br>
        • <b>W₁₈</b> = จำนวน ESAL ตลอดอายุออกแบบ (18-kip Equivalent Single Axle Load)<br>
        • <b>Z_R</b> = ค่า Standard Normal Deviate ที่ระดับความน่าเชื่อถือ R<br>
        • <b>S₀</b> = Overall Standard Deviation (0.40–0.50 สำหรับผิวทางยาง)<br>
        • <b>SN</b> = Structural Number (ค่าที่ต้องหา)<br>
        • <b>ΔPSI</b> = การเปลี่ยนแปลง PSI = pᵢ - pₜ<br>
        • <b>M_R</b> = Resilient Modulus ของดินพื้นทาง (psi)<br>
        • <b>a₁, a₂, a₃</b> = Layer Coefficients<br>
        • <b>D₁, D₂, D₃</b> = ความหนาชั้นทาง (นิ้ว)<br>
        • <b>m₂, m₃</b> = Drainage Coefficients<br>
        </div>
        """, unsafe_allow_html=True)
    
    with col_i2:
        st.markdown("""
        <div class="info-box">
        <b style="color:#a8d0f0;">ขั้นตอนการออกแบบ AASHTO 1993:</b><br>
        1. กำหนด W₁₈ จากการสำรวจจราจรและ Load Equivalency Factor (LEF)<br>
        2. เลือกระดับความน่าเชื่อถือ (R) ตามประเภทถนน<br>
        3. กำหนด S₀, pᵢ, pₜ<br>
        4. วัดหรือประมาณค่า M_R ของดินพื้นทาง<br>
        5. แก้สมการหาค่า SN ที่ต้องการ<br>
        6. เลือกวัสดุและ Layer Coefficients<br>
        7. กำหนด Drainage Coefficients<br>
        8. ออกแบบความหนาแต่ละชั้น (D₁, D₂, D₃)<br>
        9. ตรวจสอบ SN ที่ได้ ≥ SN ที่ต้องการ<br>
        10. ตรวจสอบความหนาขั้นต่ำของแต่ละชั้น
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:12px;">
    <b style="color:#a8d0f0;">อ้างอิง / References:</b><br>
    • AASHTO (1993). <i>AASHTO Guide for Design of Pavement Structures.</i> American Association of State Highway and Transportation Officials, Washington D.C.<br>
    • กรมทางหลวง (2566). <i>มาตรฐานการออกแบบโครงสร้างชั้นทาง.</i> กระทรวงคมนาคม.<br>
    • Huang, Y.H. (2004). <i>Pavement Analysis and Design, 2nd Ed.</i> Pearson Prentice Hall.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border-color:#1e3a5f; margin:32px 0 16px 0;">
<div style="text-align:center; color:#2d5a8a; font-size:0.8rem; font-family:'Sarabun',sans-serif;">
    🛣️ กรมทางหลวง — Department of Highways Thailand &nbsp;|&nbsp; AASHTO 1993 Flexible Pavement Design Tool<br>
    <span style="font-size:0.7rem; color:#1e3a5f;">สร้างด้วย Streamlit + Python | สำหรับการศึกษาและประกอบการออกแบบเบื้องต้นเท่านั้น</span>
</div>
""", unsafe_allow_html=True)
