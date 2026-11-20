"""CardioScope dashboard (Streamlit).

Run from the project folder:   streamlit run streamlit_app.py
Reads the artifacts written by CardioScope_EndToEnd.ipynb and data/raw/framingham.csv.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from torch import nn

ROOT = Path(__file__).resolve().parent
ART, RAW = ROOT / "artifacts", ROOT / "data" / "raw" / "framingham.csv"
TARGET = "TenYearCHD"

st.set_page_config(page_title="CardioScope", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

if not (ART / "model_card.json").exists() or not RAW.exists():
    st.error("Artifacts not found. Run every cell of `CardioScope_EndToEnd.ipynb` first "
             "(it trains the models and writes `artifacts/`), and keep `data/raw/framingham.csv` in place.")
    st.stop()

import plotly.io as pio
pio.templates["cardio"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["cardio"].layout.font = dict(color="#111827")
pio.templates["cardio"].layout.paper_bgcolor = "rgba(0,0,0,0)"
pio.templates["cardio"].layout.plot_bgcolor = "rgba(0,0,0,0)"
pio.templates.default = "cardio"

# ====================================================================== palette & style
VIVID = ["#4361ee", "#f72585", "#4cc9f0", "#fb8500", "#2a9d8f", "#7209b7", "#ffbe0b", "#e5383b"]
OUTCOME = {"No CHD": "#4cc9f0", "CHD": "#f72585"}
RISK_COL = {"low": "#2ecc71", "moderate": "#f5b700", "high": "#e5383b"}
# one multi-colour scale for "risk %" everywhere (green -> yellow -> orange -> red -> purple)
RISK_SCALE = [[0, "#2ecc71"], [10 / 60, "#f5b700"], [20 / 60, "#fb8500"], [35 / 60, "#e5383b"], [1, "#7209b7"]]
DOT = {"tn": "#a8dadc", "fp": "#fb8500", "fn": "#e5383b", "tp": "#2ecc71"}
EDU = {1: "Some high school", 2: "High school / GED", 3: "Some college", 4: "College degree"}

CSS = """
<style>
/* force a light surface with dark text everywhere, whatever theme the browser or Streamlit picks */
:root{color-scheme:light}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#ffffff!important;color:#111827!important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#f4f6fb!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,.stApp p,.stApp li,.stApp label,.stApp span,.stApp small,.stApp div,
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stMetricLabel"] *,.stApp [data-testid="stMetricValue"] *,.stApp button[data-baseweb="tab"] *,
.stApp [data-testid="stExpander"] summary *,.stApp [data-baseweb="radio"] *,.stApp [data-testid="stToggle"] *{color:#111827}
.stApp button[data-baseweb="tab"][aria-selected="true"] *{color:#4361ee}
.stApp input,.stApp textarea,.stApp [data-baseweb="select"] *{color:#111827!important;background-color:#ffffff}
.stApp [data-baseweb="popover"] *{color:#111827!important;background:#ffffff}
.stApp .stButton button,.stApp [data-testid="stBaseButton-secondary"]{background:#ffffff;color:#111827!important;border:1px solid #cfd6e4}
.stApp [data-testid="stCaptionContainer"],.stApp small{color:#4b5563!important}
.stApp .hero,.stApp .hero *,.stApp .step,.stApp .step *,.stApp .badge{color:#ffffff!important}
.stApp .mcard .auc{color:var(--c)!important}
.stApp [data-testid="stMetric"],.stApp .mcard{background:#f1f4fa}
.stApp [data-testid="stAlert"] *{color:inherit}
.block-container{padding-top:4.2rem!important;max-width:1260px}
[data-testid="stHeader"]{background:#ffffff!important;height:3rem}
[data-baseweb="tab-list"]{background:#ffffff!important;position:relative;z-index:2;border-bottom:1px solid #e5e7eb}
.stApp *{color:#111827!important}
.stApp small,.stApp [data-testid="stCaptionContainer"] *{color:#4b5563!important}
.stApp button[data-baseweb="tab"][aria-selected="true"],.stApp button[data-baseweb="tab"][aria-selected="true"] *{color:#4361ee!important}
.stApp .hero *,.stApp .step *,.stApp .badge{color:#ffffff!important}
.stApp .mcard .auc{color:var(--c)!important}
.stApp [data-testid^="stAlert"],.stApp [data-testid^="stAlert"] *{color:#111827!important}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes grad{0%{background-position:0% 50%}100%{background-position:100% 50%}}
@keyframes scan{to{stroke-dashoffset:-.3}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(46,204,113,.55)}70%,100%{box-shadow:0 0 0 9px rgba(46,204,113,0)}}
.hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 32px;margin-bottom:18px;color:#fff;
  background:linear-gradient(120deg,#4361ee,#7209b7,#f72585,#fb8500);background-size:300% 300%;animation:grad 14s ease infinite alternate, fadeUp .7s both;
  box-shadow:0 14px 40px rgba(67,97,238,.28)}
.hero h1{margin:0;font-size:2.15rem;letter-spacing:-.02em;color:#fff;padding:0}
.hero p{margin:.5rem 0 0;max-width:46ch;color:rgba(255,255,255,.9);font-size:1.02rem}
.hero .eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;opacity:.85}
.hero .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.hero .chip{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.3);border-radius:99px;padding:4px 13px;font-size:.82rem;backdrop-filter:blur(4px)}
.hero svg{position:absolute;right:0;bottom:0;width:55%;height:80%;opacity:.95;pointer-events:none}
.hero svg path{fill:none;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.ecg-b{stroke:rgba(255,255,255,.22);stroke-width:2}
.ecg-l{stroke:#fff;stroke-width:3;stroke-dasharray:.16 1.2;stroke-dashoffset:1.1;animation:scan 3.2s linear infinite;filter:drop-shadow(0 0 6px #fff)}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border-radius:16px;padding:14px 18px;border-left:5px solid #4361ee;animation:fadeUp .6s both;transition:transform .25s}
[data-testid="stMetric"]:hover{transform:translateY(-3px)}
div[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"]{border-left-color:#f72585}
div[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"]{border-left-color:#fb8500}
div[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"]{border-left-color:#2a9d8f}
[data-baseweb="tab-panel"]{animation:fadeUp .5s both}
button[data-baseweb="tab"]{font-size:1rem;font-weight:600}
.stApp .pill-low{background:#d8f3e3;color:#1e7a45!important}
.stApp .pill-moderate{background:#fdf0c8;color:#8a5a00!important}
.stApp .pill-high{background:#fbd9d6;color:#a31f1a!important}
.pill{display:inline-block;padding:6px 18px;border-radius:99px;font-weight:700;font-size:.95rem}
.mcard{border-radius:16px;padding:16px 18px;background:rgba(128,128,128,.08);border-top:6px solid var(--c);animation:fadeUp .6s both;transition:transform .25s,box-shadow .25s}
.mcard:hover{transform:translateY(-4px);box-shadow:0 10px 26px rgba(0,0,0,.14)}
.mcard b{font-size:.98rem}.mcard .auc{font-size:2.1rem;font-weight:800;letter-spacing:-.03em;color:var(--c);line-height:1.15}
.mcard small{opacity:.75}.badge{background:var(--c);color:#fff;border-radius:99px;padding:2px 9px;font-size:.68rem;font-weight:700;margin-left:6px}
.step{border-radius:16px;padding:16px;color:#fff;min-height:132px;animation:fadeUp .7s both;transition:transform .25s}
.step:hover{transform:translateY(-5px) rotate(-.4deg)}
.step .n{width:30px;height:30px;border-radius:50%;background:rgba(255,255,255,.25);display:grid;place-items:center;font-weight:800;margin-bottom:8px}
.step b{display:block;font-size:1.02rem}.step span{font-size:.82rem;opacity:.92}
.live{display:inline-block;width:9px;height:9px;border-radius:50%;background:#2ecc71;margin-right:7px;animation:pulse 2s infinite}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
"""
DARK = bool(st.session_state.get("dark", False))
_css = CSS.replace("color:#ffffff!important", "color:@W@!important")
if DARK:
    for a, b in [("#111827", "#f3f4f6"), ("#4b5563", "#a3adc2"), ("#f4f6fb", "#161b26"), ("#f1f4fa", "#1b2230"), ("#e5e7eb", "#2d3646"),
                 ("#cfd6e4", "#3a4458"), ("color-scheme:light", "color-scheme:dark"), ("#ffffff", "#0e1117"), ("#4361ee!important", "#8ea2ff!important")]:
        _css = _css.replace(a, b)
    _css = _css.replace("</style>", """.stApp [data-testid^="stAlert"],.stApp [data-testid^="stAlert"] *{color:#111827!important}
.stApp pre,.stApp pre *,.stApp code{background:#1b2230!important;color:#f3f4f6!important}
.stApp [data-testid="stExpander"] details{background:#161b26;border-color:#2d3646}
.stApp [data-baseweb="select"]>div,.stApp input{background-color:#1b2230!important}
.stApp .stButton button{background:#1b2230!important}
.stApp [data-testid^="stNumberInputStep"],.stApp [data-testid^="stNumberInputStep"] *{background:#1b2230!important;color:#f3f4f6!important}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] div,.stApp [data-baseweb="select"] > div > div{background:#1b2230!important}
.stApp [data-testid="stExpander"] summary,.stApp [data-testid="stExpander"] summary *{background:#161b26!important}
.stApp details summary{background:#161b26!important}
.stApp [data-testid="stNumberInputContainer"],.stApp [data-testid="stNumberInputContainer"] *,.stApp [data-testid="stSelectbox"] [data-baseweb="select"] *,.stApp [data-testid="stSelectbox"] [data-baseweb="select"]{background-color:#1b2230!important}
.stApp [data-baseweb="select"] div,.stApp [data-testid="stSelectbox"] [data-baseweb="select"]>div>div,.stApp [data-testid="stSelectbox"] [data-baseweb="select"]>div>div *{background-color:#1b2230!important}
.stApp [data-testid="stSelectbox"]>div:last-child,.stApp [data-testid="stSelectbox"]>div:last-child div,.stApp [data-testid="stCode"] *,.stApp [data-testid^="stElementToolbar"] *{background-color:#1b2230!important}
.stApp input::placeholder{color:#a3adc2!important;opacity:1!important;-webkit-text-fill-color:#a3adc2!important}
.stApp [data-baseweb="select"] [aria-live],.stApp [data-baseweb="select"] [class*="placeholder"]{color:#a3adc2!important}
.stApp [data-testid="stNumberInputContainer"] svg,.stApp [data-baseweb="select"] svg{fill:#f3f4f6!important;color:#f3f4f6!important}
.stApp button[data-variant="segmented_control"],.stApp button[data-variant="pills"]{background:#1b2230!important;border-color:#3a4458!important}
.stApp button[data-variant="segmented_control"][aria-checked="true"],.stApp button[data-variant="pills"][aria-checked="true"],.stApp button[data-variant="pills"][aria-pressed="true"]{background:#2d3a63!important;border-color:#8ea2ff!important}
.stApp [data-baseweb="select"] div[aria-hidden="true"],.stApp [data-baseweb="select"] svg{background:#1b2230!important}
.stApp [data-baseweb="input"],.stApp [data-baseweb="input"] *,.stApp [data-baseweb="base-input"],.stApp [data-baseweb="select"] *{background-color:#1b2230!important}
.stApp [data-baseweb="input"] svg,.stApp [data-baseweb="select"] svg{fill:#f3f4f6!important}
.stApp [data-testid^="stBaseButton-segmented_control"],.stApp [data-testid^="stBaseButton-pills"]{background:#1b2230!important;border-color:#3a4458!important}
.stApp [data-testid="stBaseButton-segmented_controlActive"],.stApp [data-testid="stBaseButton-pillsActive"]{background:#2d3a63!important;border-color:#8ea2ff!important}
.stApp [data-testid="stSlider"] [role="slider"]{background:#8ea2ff!important}
</style>""")

# widget skin: set explicitly for BOTH modes so it never depends on the host's Streamlit theme
_T = dict(IN_BG="#1b2230", IN_FG="#f3f4f6", BORDER="#3a4458", TRACK="#4a5570", ACC="#8ea2ff", PH="#a3adc2", SEG="#2d3a63") if DARK else \
     dict(IN_BG="#ffffff", IN_FG="#111827", BORDER="#aab4c8", TRACK="#c3cad8", ACC="#4361ee", PH="#6b7280", SEG="#e0e7ff")
_W = """
.stApp [data-testid="stNumberInputContainer"],.stApp [data-testid="stNumberInputContainer"] *,
.stApp [data-baseweb="input"],.stApp [data-baseweb="input"] *,.stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"],.stApp [data-baseweb="select"] div,
.stApp [data-testid="stSelectbox"]>div:last-child,.stApp [data-testid="stSelectbox"]>div:last-child div{background-color:@IN_BG@!important;color:@IN_FG@!important}
.stApp [data-testid="stNumberInputContainer"],.stApp [data-baseweb="select"]>div{border-color:@BORDER@!important}
.stApp [data-testid^="stNumberInputStep"],.stApp [data-testid^="stNumberInputStep"] *{background-color:@IN_BG@!important;color:@IN_FG@!important}
.stApp [data-testid="stNumberInputContainer"] svg,.stApp [data-baseweb="select"] svg,.stApp [data-baseweb="input"] svg{fill:@IN_FG@!important;color:@IN_FG@!important}
.stApp input::placeholder{color:@PH@!important;opacity:1!important;-webkit-text-fill-color:@PH@!important}
.stApp [data-baseweb="select"] [aria-live],.stApp [data-baseweb="select"] [class*="placeholder"]{color:@PH@!important}
.stApp [data-baseweb="popover"],.stApp [data-baseweb="popover"] *,.stApp [data-baseweb="menu"],.stApp [role="listbox"],.stApp [role="option"]{background-color:@IN_BG@!important;color:@IN_FG@!important}
.stApp .stButton button,.stApp [data-testid^="stBaseButton-secondary"]{background-color:@IN_BG@!important;color:@IN_FG@!important;border:1px solid @BORDER@!important}
.stApp button[data-variant="segmented_control"],.stApp button[data-variant="pills"]{background-color:@IN_BG@!important;color:@IN_FG@!important;border:1px solid @BORDER@!important}
.stApp button[data-variant="segmented_control"][aria-checked="true"],.stApp button[data-variant="pills"][aria-checked="true"],.stApp button[data-variant="pills"][aria-pressed="true"],
.stApp button[data-variant="segmented_control"][aria-pressed="true"]{background-color:@SEG@!important;border-color:@ACC@!important}
.stApp [data-testid="stCheckbox"] label>span+div{background-color:@TRACK@!important}
.stApp [data-testid="stCheckbox"] label>span+div>div{background-color:#ffffff!important}
.stApp [data-testid="stCheckbox"] label:has(input:checked)>span+div{background-color:@ACC@!important}
.stApp [data-testid="stRadioOption"]>span+div>div:first-child{background-color:@IN_BG@!important;border:2px solid @TRACK@!important}
.stApp [data-testid="stRadioOption"]:has(input:checked)>span+div>div:first-child{background-color:@ACC@!important;border-color:@ACC@!important}
.stApp [data-testid="stRadioOption"]:has(input:checked)>span+div>div:first-child>div{background-color:#ffffff!important}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"],.stApp [data-testid="stSelectbox"] [data-baseweb="select"]>div{border:1px solid @BORDER@!important;border-radius:8px}
.stApp [data-baseweb="radio"]>div:first-child{background-color:@IN_BG@!important;border:2px solid @TRACK@!important}
.stApp [data-baseweb="radio"] input:checked+div,.stApp [data-baseweb="radio"]:has(input:checked)>div:first-child{background-color:@ACC@!important;border-color:@ACC@!important}
.stApp [data-testid="stAlertContainer"]{background-color:#eef2f7!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){background-color:#dcf5e5!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){background-color:#fdf0c8!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){background-color:#fbd9d6!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]){background-color:#dbe8fd!important}
.stApp [data-testid="stAlertContainer"],.stApp [data-testid="stAlertContainer"] *{color:#111827!important}
.stApp [data-testid="stExpander"] details,.stApp [data-testid="stExpander"] summary{background-color:@IN_BG@!important;border-color:@BORDER@!important}
.stApp [data-testid="stSlider"] [role="slider"]{background-color:@ACC@!important}
"""
for k, v in _T.items():
    _W = _W.replace("@" + k + "@", v)
_css = _css.replace("</style>", _W + "</style>")
st.markdown(_css.replace("@W@", "#ffffff"), unsafe_allow_html=True)

# plotly figures are cached, so the light/dark skin is applied at draw time
from streamlit.delta_generator import DeltaGenerator
if not getattr(DeltaGenerator, "_cardio_patched", False):
    _orig_pc = DeltaGenerator.plotly_chart

    def _themed_pc(self, fig, *a, **k):
        dark = bool(st.session_state.get("dark", False))
        fig = go.Figure(fig)
        ink = "#f3f4f6" if dark else "#111827"
        fig.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=ink)
        if len(fig.layout.updatemenus):
            fig.update_layout(updatemenus=[dict(m.to_plotly_json(), bgcolor="#1b2230" if dark else "#f1f4fa",
                                                bordercolor="#3a4458" if dark else "#cfd6e4", font=dict(color=ink)) for m in fig.layout.updatemenus])
        fig.update_traces(selector=dict(type="indicator"), number_font_color=ink, title_font_color=ink, gauge_axis_tickfont_color=ink)
        return _orig_pc(self, fig, *a, **k)

    DeltaGenerator.plotly_chart = _themed_pc
    st.plotly_chart = lambda *a, **k: st._main.plotly_chart(*a, **k)
    DeltaGenerator._cardio_patched = True


# ====================================================================== portable inference
# (identical to the notebook's section 9: no custom pickles, so any program can load the models)
class _Net(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x).squeeze(-1)


PRM = json.loads((ART / "preprocessing.json").read_text())
CARD = json.loads((ART / "model_card.json").read_text())
FEATURES, CONTINUOUS = PRM["features"], PRM["continuous"]
LABELS, MODEL_COLORS = CARD["labels"], CARD["model_colors"]
NAMES = list(CARD["model_files"])
BASE, BANDS, REF = CARD["base_rate"], CARD["risk_bands"], CARD["reference_profile"]


def to_frame(X):
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    for col in FEATURES:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[FEATURES].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def prep_transform(X, prm):
    X = to_frame(X)
    X["glucose_missing"] = X["glucose"].isna().astype(int)
    X.loc[(X["currentSmoker"] == 0) & X["cigsPerDay"].isna(), "cigsPerDay"] = 0.0
    X.loc[(X["currentSmoker"] == 1) & X["cigsPerDay"].isna(), "cigsPerDay"] = prm["smoker_cigs_median"]
    for col, val in {**prm["medians"], **prm["modes"]}.items():
        X[col] = X[col].fillna(val)
    cont = prm["continuous"]
    X[cont] = (X[cont] - np.array(prm["scaler_mean"])) / np.array(prm["scaler_scale"])
    return X[prm["features_out"]].to_numpy(np.float32)


class PortableModel:
    def __init__(self, name, art=ART):
        self.prm = json.loads((art / "preprocessing.json").read_text())
        path = art / CARD["model_files"][name]
        self.kind = "torch" if path.suffix == ".pt" else "sk"
        if self.kind == "torch":
            blob = torch.load(path, map_location="cpu")
            self.net = _Net(blob["n_in"], tuple(blob["hidden"]), blob["dropout"])
            self.net.load_state_dict(blob["state_dict"])
            self.net.eval()
        elif path.suffix == ".json":
            import xgboost as xgb
            self.clf = xgb.XGBClassifier()
            self.clf.load_model(path)
        else:
            self.clf = joblib.load(path)

    def predict_proba(self, X):
        Z = prep_transform(X, self.prm)
        if self.kind == "torch":
            with torch.no_grad():
                p = torch.sigmoid(self.net(torch.from_numpy(Z))).numpy().astype(np.float64)
            return np.column_stack([1 - p, p])
        return self.clf.predict_proba(Z)


GROUPS = {"Smoking": ["currentSmoker", "cigsPerDay"]}


def local_factors(model, frame, top_k=6):
    """Swap each factor to the typical value one at a time; positive = raises risk vs a typical patient."""
    grouped = {f for fs in GROUPS.values() for f in fs}
    factors = {LABELS[f]: [f] for f in FEATURES if f not in grouped}
    factors.update(GROUPS)
    base = model.predict_proba(frame)[:, 1]
    names, deltas = list(factors), np.zeros(len(factors))
    for j, label in enumerate(names):
        swapped = frame.copy()
        for f in factors[label]:
            swapped[f] = REF[f]
        deltas[j] = base[0] - model.predict_proba(swapped)[0, 1]
    out = []
    for j in np.argsort(-np.abs(deltas))[:top_k]:
        if abs(deltas[j]) > 1e-4:
            out.append({"factor": names[j], "change": float(deltas[j]) * 100,
                        "you": ", ".join(f"{f}={frame.iloc[0][f]:g}" for f in factors[names[j]] if pd.notna(frame.iloc[0][f])) or "not given",
                        "typical": ", ".join(f"{f}={REF[f]:g}" for f in factors[names[j]])})
    return out


def cat_of(p):
    return "low" if p < BANDS["low_below"] else "high" if p >= BANDS["high_from"] else "moderate"


# ====================================================================== cached loaders
@st.cache_data
def load_data():
    return pd.read_csv(RAW).drop_duplicates().reset_index(drop=True).apply(pd.to_numeric, errors="coerce")


@st.cache_data
def load_json(name):
    return json.loads((ART / name).read_text())


@st.cache_data
def load_preds():
    return pd.read_csv(ART / "test_predictions.csv")


@st.cache_resource
def get_model(name):
    return PortableModel(name)


slug = lambda n: n.lower().replace(" ", "_").replace("(", "").replace(")", "")


def style(fig, h=380, legend=True):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=44, b=8), font_size=13, title_font_size=16,
                      legend=dict(orientation="h", y=-0.18) if legend else dict(visible=False))
    return fig


EXPLAIN = {
    'gauge': ('Risk gauge', "The needle shows the model's estimated chance of coronary heart disease within 10 years for the profile you entered. Background bands run green (low) to yellow, orange, red and purple (very high). The blue marker is the average risk in the study cohort, so a needle past it means above-average risk."),
    'factors': ('What moves this result', 'Each bar shows how much one factor changes your risk compared with a typical patient, in percentage points. Red bars push risk up, green bars pull it down, and the longest bars matter most. It is found by swapping one factor at a time to the typical value and re-scoring, so it describes this model, not cause and effect.'),
    'cmp': ('Same patient, four models', 'The same profile scored by all four models. If the bars are close, the models agree and the estimate is stable. A large gap means the profile sits where the models extrapolate differently, so treat the number with more caution.'),
    'sweep_chart': ('Risk curve', 'Only the selected factor changes along the x-axis while everything else stays as you entered it. Marker colour follows risk (green low to purple high), the shaded bands match low / moderate / high, and the star marks you. A steep section shows where a small change matters most; a flat one means little effect.'),
    'scen_chart': ('What-if bars', "Each bar is your estimated risk after one change, such as stopping smoking or lowering blood pressure, with the change in points beside it. 'All of the above' combines every change. These are model associations, not a promise that the change will cut your risk by that much."),
    'donut': ('Outcome split', 'Share of patients in the dataset who did or did not develop CHD within 10 years. It shows how imbalanced the problem is: far fewer people have the outcome than not, which is why accuracy alone is a poor yardstick.'),
    'mult': ('Risk multipliers', 'Each bar compares the CHD rate of a higher-risk group with a lower-risk group, such as age 65+ versus 30-39. A value of 3× means the rate is three times as high. Hover to see the two underlying rates. These are raw group comparisons and do not adjust for other factors.'),
    'rate': ('CHD rate by group', 'Patients are grouped by the chosen factor and each bar is the share who developed CHD. The dashed line is the overall rate, so bars above it are higher-risk groups. Hover for the group size, because small groups give noisy rates.'),
    'heat': ('Correlation heatmap', 'Each cell is the correlation (r) between two variables, from -1 (move in opposite directions) to +1 (move together); 0 means no linear link. Variables are sorted by their link with the 10-year CHD outcome. Strongly related pairs, like systolic and diastolic pressure, carry overlapping information.'),
    'hist': ('Distribution by outcome', 'Overlaid histograms show how a measurement is spread among patients with and without CHD, as a percentage of each group. The box plot above shows the median and spread. The more the two shapes shift apart, the more useful that variable is for telling the groups apart.'),
    'ratemap': ('Two-factor risk map', 'Patients are split into seven bands on each of two factors and each cell shows the CHD rate for that combination. Warm colours mark where risk concentrates, and cells with fewer than 15 patients are left blank because their rates are unreliable.'),
    'violins': ('Violin plots', 'Each violin is the distribution of one measurement: wider parts mean more patients at that value. The inner box shows the median and quartiles and the line marks the mean. Long tails point to outliers, such as very high cholesterol or glucose.'),
    'missing': ('Missing data', "Bars show the percentage of patients with no recorded value for each measurement. Glucose is usually the largest gap, which is why the pipeline adds a 'glucose missing' flag and fills the rest with the median instead of dropping rows."),
    'picto': ('100-patient picture', 'The test results rescaled to 100 people. Each dot is a patient: caught (CHD, flagged), missed (CHD, not flagged), false alarm (healthy, flagged) or correctly cleared. It turns the confusion matrix into counts you can picture.'),
    'conf': ('Confusion matrix', "Rows are the true outcome and columns are the model's prediction on the test set. The top-left and bottom-right cells are correct calls. Top-right is false alarms and bottom-left is missed CHD cases. Sensitivity and specificity are computed from these four counts."),
    'meters': ('Detection quality', 'Sensitivity is the share of real CHD cases caught. Specificity is the share of healthy people correctly cleared. Precision is how many flagged people truly develop CHD, and F1 balances precision and recall. Higher is better, but they trade off against each other.'),
    'thrfig': ('Threshold trade-off', 'Each line is a metric at every possible decision threshold; the red line is the one chosen with the slider. Lowering the threshold raises sensitivity and the share flagged, but lowers specificity and precision. Choose the point that suits the cost of a missed case versus a false alarm.'),
    'curves': ('ROC, precision-recall and calibration', "ROC: curves closer to the top-left are better and the dashed diagonal is a coin flip. Precision-recall: higher is better, which matters for the rarer CHD class. Calibration: points near the dashed diagonal mean predicted risk matches the observed rate, so '20%' really means about 20 in 100."),
    'auc': ('AUC: cross-validation vs test', 'Solid bars are the cross-validated AUC used to choose the model and hatched bars are the held-out test AUC. Similar heights mean the model generalises. Gaps of about 0.03 are expected from test-set noise and are not proof that one model is better.'),
    'radar': ('Model profile', "Each model's shape across five scores, with the centre at 0.5 (a coin flip). A bigger, more even shape means better all-round performance, and a lopsided one means the model is strong at one thing, such as sensitivity, at the cost of another."),
    'imp': ('Feature importance', 'Permutation importance: each feature is shuffled in turn and the bar shows how much the test AUC drops. A longer bar means the model relies on that feature more. Error bars show variation across repeats, and a bar near zero means the feature adds little.'),
    'ann': ('ANN training curves', 'Left: training loss (blue) versus validation loss (pink) per epoch, with the dotted line marking the epoch kept by early stopping. If validation loss rises while training loss keeps falling, the network is overfitting. Right: validation AUC over the same epochs.'),
}


def explain(key, box=st):
    title, text = EXPLAIN[key]
    with box.expander(f"📘 How to read: {title}"):
        st.write(text)


# ====================================================================== figure builders
def gauge_fig(p):
    c = RISK_COL[cat_of(p)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 56}},
        title={"text": "10-year CHD risk", "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 10, 20, 35, 60, 100], "ticksuffix": "%"},
               "bar": {"color": c, "thickness": 0.3}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 10], "color": "rgba(46,204,113,.55)"}, {"range": [10, 20], "color": "rgba(245,183,0,.6)"},
                         {"range": [20, 35], "color": "rgba(251,133,0,.6)"}, {"range": [35, 60], "color": "rgba(229,56,59,.6)"},
                         {"range": [60, 100], "color": "rgba(114,9,183,.6)"}],
               "threshold": {"line": {"color": "#4361ee", "width": 5}, "thickness": 0.92, "value": BASE * 100}}))
    fig.update_layout(height=300, margin=dict(l=24, r=24, t=50, b=8))
    return fig


def factors_fig(items):
    d = pd.DataFrame(items).sort_values("change")
    m = max(abs(d.change).max(), 1)
    fig = go.Figure(go.Bar(
        x=d.change, y=d.factor, orientation="h", text=[f"{v:+.1f}" for v in d.change], textposition="outside",
        marker=dict(color=d.change, colorscale="RdYlGn_r", cmin=-m, cmax=m, line=dict(width=0)),
        customdata=np.stack([d.you, d.typical], axis=1),
        hovertemplate="<b>%{y}</b><br>You: %{customdata[0]}<br>Typical: %{customdata[1]}<br>Risk change: %{x:+.1f} pts<extra></extra>"))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_xaxes(title="change in risk vs a typical patient (percentage points)", range=[-m * 1.35, m * 1.35])
    return style(fig, 300 + 8 * len(d), False).update_layout(title="What moves this result")


def models_fig(probs):
    d = pd.DataFrame({"model": list(probs), "risk": [v * 100 for v in probs.values()]})
    fig = go.Figure(go.Bar(x=d.model, y=d.risk, marker_color=[MODEL_COLORS[m] for m in d.model], text=[f"{v:.1f}%" for v in d.risk],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="predicted risk (%)", range=[0, max(d.risk.max() * 1.25, 12)])
    return style(fig, 300, False).update_layout(title="Same patient, four models")


SWEEPS = {
    "Age": ("age", 32, 70, 2, " yrs"), "Systolic BP": ("sysBP", 90, 220, 5, " mmHg"), "Cholesterol": ("totChol", 120, 400, 10, " mg/dL"),
    "Glucose": ("glucose", 60, 300, 10, " mg/dL"), "BMI": ("BMI", 16, 45, 1, ""), "Heart rate": ("heartRate", 45, 120, 3, " bpm"),
    "Cigarettes/day": ("cigsPerDay", 0, 60, 2, "/day"),
}


def sweep_data(model, patient, label):
    key, lo, hi, step, unit = SWEEPS[label]
    if key == "sysBP":
        lo = max(lo, int(np.ceil((patient["diaBP"] + 5) / step) * step))
    xs = list(np.arange(lo, hi + 1e-9, step))
    rows = []
    for v in xs:
        r = dict(patient)
        if key == "cigsPerDay":
            r["currentSmoker"] = int(v > 0)
            r["cigsPerDay"] = v if v > 0 else None
        else:
            r[key] = v
        rows.append(r)
    return xs, model.predict_proba(pd.DataFrame(rows))[:, 1] * 100


def sweep_fig(xs, ys, label, cur, unit):
    ymax = max(25, float(np.ceil(max(ys) * 1.12 / 5) * 5))
    fig = go.Figure()
    for lo, hi, col in [(0, 10, "rgba(46,204,113,.15)"), (10, 20, "rgba(245,183,0,.17)"), (20, 100, "rgba(229,56,59,.12)")]:
        if lo < ymax:
            fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=col, line_width=0, layer="below")
    mk = dict(size=11, color=list(ys), colorscale=RISK_SCALE, cmin=0, cmax=60, line=dict(color="white", width=1.5),
              colorbar=dict(title=dict(text="risk %", side="top"), thickness=12, len=0.7, y=0.4))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="rgba(130,130,150,.6)", width=3, shape="spline"),
                             marker=mk, fill="tozeroy", fillcolor="rgba(67,97,238,.10)", name="risk",
                             hovertemplate=f"{label}: %{{x}}{unit}<br>risk: %{{y:.1f}}%<extra></extra>"))
    if cur is not None and xs[0] <= cur[0] <= xs[-1]:
        fig.add_trace(go.Scatter(x=[cur[0]], y=[cur[1]], mode="markers+text", text=["You"], textposition="top center",
                                 marker=dict(size=20, symbol="star", color="#ffbe0b", line=dict(color="#222", width=2)),
                                 hovertemplate=f"You: {cur[0]:g}{unit}<br>{cur[1]:.1f}%<extra></extra>", showlegend=False))
    frames = [go.Frame(data=[go.Scatter(x=xs[:k], y=ys[:k], marker=dict(color=list(ys[:k]), colorscale=RISK_SCALE, cmin=0, cmax=60))], traces=[0])
              for k in range(2, len(xs) + 1)]
    fig.frames = frames
    fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.16, showactive=False, pad=dict(t=0, r=6),
                                        buttons=[dict(label="▶  Replay animation", method="animate",
                                                      args=[None, dict(frame=dict(duration=70, redraw=True), fromcurrent=False, transition=dict(duration=0))])])])
    fig.update_xaxes(title=label + (f" ({unit.strip()})" if unit.strip() else ""))
    fig.update_yaxes(title="10-year risk (%)", range=[0, ymax])
    return style(fig, 410, False)


def scenarios_data(model, patient, p):
    mods = []
    if patient["currentSmoker"]:
        mods.append(("Stop smoking", {"currentSmoker": 0, "cigsPerDay": None}))
    if patient["sysBP"] > 120:
        mods.append(("Blood pressure to 120/80", {"sysBP": 120, "diaBP": min(patient["diaBP"], 80)}))
    if (patient.get("totChol") or 0) > 200:
        mods.append(("Cholesterol to 200", {"totChol": 200}))
    if (patient.get("glucose") or 0) > 100:
        mods.append(("Glucose to 90", {"glucose": 90}))
    if (patient.get("BMI") or 0) > 25:
        mods.append(("BMI to 24", {"BMI": 24}))
    if len(mods) > 1:
        allm = {}
        for _, m in mods:
            allm.update(m)
        mods.append(("All of the above", allm))
    if not mods:
        return []
    rows = [{**patient, **m} for _, m in mods]
    probs = model.predict_proba(pd.DataFrame(rows))[:, 1]
    return sorted([(n, float(r)) for (n, _), r in zip(mods, probs)], key=lambda t: t[1])


def scenarios_fig(rows, p):
    labels = ["Current profile"] + [r[0] for r in rows]
    vals = [p * 100] + [r[1] * 100 for r in rows]
    text = [f"<b>{vals[0]:.1f}%</b>"] + [f"<b>{v:.1f}%</b> ({v - vals[0]:+.1f})" for v in vals[1:]]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", text=text, textposition="outside", cliponaxis=False,
                           marker=dict(color=vals, colorscale=RISK_SCALE, cmin=0, cmax=60, line=dict(width=0)),
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title="estimated 10-year risk (%)", range=[0, max(vals) * 1.75])
    fig.update_yaxes(autorange="reversed", automargin=True)
    return style(fig, 120 + 52 * len(labels), False).update_layout(margin=dict(l=8, r=40, t=10, b=8))


@st.cache_data
def groupings_table(label):
    df = load_data()
    cuts = {
        "Age": pd.cut(df.age, [30, 40, 45, 50, 55, 60, 65, 75], right=False, labels=["30-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]),
        "Sex": df.male.map({0: "Female", 1: "Male"}), "Smoking": df.currentSmoker.map({0: "Non-smoker", 1: "Smoker"}),
        "Hypertension": df.prevalentHyp.map({0: "No", 1: "Yes"}), "Diabetes": df.diabetes.map({0: "No", 1: "Yes"}),
        "Systolic BP": pd.cut(df.sysBP, [0, 120, 130, 140, 160, 400], right=False, labels=["<120", "120-129", "130-139", "140-159", "160+"]),
        "Cholesterol": pd.cut(df.totChol, [0, 200, 240, 1000], right=False, labels=["<200", "200-239", "240+"]),
        "BMI": pd.cut(df.BMI, [0, 18.5, 25, 30, 100], right=False, labels=["<18.5", "18.5-24.9", "25-29.9", "30+"]),
        "Education": df.education.map(EDU),
    }
    g = df.groupby(cuts[label], observed=True)[TARGET].agg(["mean", "count"]).reset_index()
    g.columns = ["group", "rate", "n"]
    g["rate"] *= 100
    g["group"] = g["group"].astype(str)
    return g


RATE_LABELS = ["Age", "Sex", "Smoking", "Hypertension", "Diabetes", "Systolic BP", "Cholesterol", "BMI", "Education"]


@st.cache_data
def rate_fig(label):
    g, avg = groupings_table(label), load_data()[TARGET].mean() * 100
    fig = go.Figure(go.Bar(x=g.group, y=g.rate, text=[f"{v:.1f}%" for v in g.rate], textposition="outside",
                           marker=dict(color=g.rate, colorscale="RdYlBu_r", cmid=avg, cmin=0, cmax=max(g.rate.max(), avg * 2),
                                       colorbar=dict(title="CHD %", thickness=12), line=dict(width=0)),
                           customdata=g.n, hovertemplate="%{x}<br>%{y:.1f}% developed CHD<br>%{customdata:,} patients<extra></extra>"))
    fig.add_hline(y=avg, line_dash="dash", line_color="#444", annotation_text=f"overall {avg:.1f}%", annotation_position="top left")
    fig.update_yaxes(title="10-year CHD rate (%)", range=[0, g.rate.max() * 1.2])
    return style(fig, 420, False)


@st.cache_data
def multipliers_fig():
    pairs = [("Age 65+ vs 30–39", "Age", "65+", "30-39"), ("Systolic BP 160+ vs <120", "Systolic BP", "160+", "<120"),
             ("Diabetes vs none", "Diabetes", "Yes", "No"), ("Hypertension vs none", "Hypertension", "Yes", "No"),
             ("Cholesterol 240+ vs <200", "Cholesterol", "240+", "<200"), ("BMI 30+ vs healthy", "BMI", "30+", "18.5-24.9"),
             ("Male vs female", "Sex", "Male", "Female"), ("Smoker vs non-smoker", "Smoking", "Smoker", "Non-smoker")]
    rows = []
    for name, lab, hi, lo in pairs:
        t = groupings_table(lab).set_index("group")
        rows.append((name, t.loc[hi, "rate"] / t.loc[lo, "rate"], t.loc[hi, "rate"], t.loc[lo, "rate"]))
    d = pd.DataFrame(rows, columns=["name", "x", "hi", "lo"]).sort_values("x")
    fig = go.Figure(go.Bar(x=d.x, y=d.name, orientation="h", text=[f"{v:.1f}×" for v in d.x], textposition="outside",
                           marker=dict(color=d.x, colorscale="Turbo", line=dict(width=0)), customdata=np.stack([d.hi, d.lo], axis=1),
                           hovertemplate="%{y}<br>%{customdata[0]:.1f}% vs %{customdata[1]:.1f}%<extra></extra>"))
    fig.update_xaxes(title="how many times higher the CHD rate is", range=[0, d.x.max() * 1.18])
    return style(fig, 420, False)


@st.cache_data
def donut_fig():
    df = load_data()
    c = df[TARGET].value_counts().sort_index()
    fig = go.Figure(go.Pie(labels=["No CHD", "CHD"], values=c.values, hole=0.66, sort=False, direction="clockwise", rotation=0,
                           marker=dict(colors=[OUTCOME["No CHD"], OUTCOME["CHD"]], line=dict(color="white", width=3)), textinfo="label+value",
                           hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{df[TARGET].mean():.1%}</b><br>developed CHD", showarrow=False, font=dict(size=22))
    return style(fig, 380, False)


@st.cache_data
def heat_fig():
    df = load_data()
    corr = df[FEATURES + [TARGET]].corr()
    order = corr[TARGET].abs().sort_values(ascending=False).index
    c = corr.loc[order, order]
    names = [LABELS.get(o, "10-year CHD") for o in order]
    fig = px.imshow(c.values, x=names, y=names, color_continuous_scale="Spectral_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_traces(textfont_size=10, hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>")
    fig.update_xaxes(tickangle=-40)
    return style(fig, 640, False).update_layout(coloraxis_colorbar=dict(title="r", thickness=14))


@st.cache_data
def violins_fig():
    df = load_data()
    fig = make_subplots(rows=2, cols=4, subplot_titles=[LABELS[c] for c in CONTINUOUS], vertical_spacing=0.2, horizontal_spacing=0.09)
    for i, col in enumerate(CONTINUOUS):
        fig.add_trace(go.Violin(y=df[col].dropna(), name=LABELS[col], box_visible=True, meanline_visible=True, fillcolor=VIVID[i % 8],
                                line_color="#8a93a6", opacity=0.8, showlegend=False), row=i // 4 + 1, col=i % 4 + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_annotations(font_size=13)
    return style(fig, 520, False).update_layout(margin=dict(l=8, r=8, t=36, b=8))


@st.cache_data
def hist_fig(col):
    df = load_data().dropna(subset=[col]).assign(outcome=lambda d: d[TARGET].map({0: "No CHD", 1: "CHD"}))
    fig = px.histogram(df, x=col, color="outcome", barmode="overlay", marginal="box", nbins=40, opacity=0.68,
                       color_discrete_map=OUTCOME, histnorm="percent")
    fig.update_xaxes(title=LABELS[col]); fig.update_yaxes(title="% of patients", row=1, col=1)
    return style(fig, 420)


@st.cache_data
def rate_map_fig(xc, yc):
    df = load_data().dropna(subset=[xc, yc])
    q = lambda s: pd.qcut(s, 7, duplicates="drop")
    d = df.assign(xb=q(df[xc]), yb=q(df[yc]))
    rate = d.pivot_table(index="yb", columns="xb", values=TARGET, aggfunc="mean", observed=True) * 100
    cnt = d.pivot_table(index="yb", columns="xb", values=TARGET, aggfunc="count", observed=True)
    rate = rate.where(cnt >= 15)
    lab = lambda idx: [f"{iv.left:.0f}–{iv.right:.0f}" for iv in idx]
    fig = go.Figure(go.Heatmap(z=rate.values, x=lab(rate.columns), y=lab(rate.index), colorscale="Turbo", zmin=0, zmax=45,
                               text=np.round(rate.values, 0), texttemplate="%{text:.0f}", customdata=cnt.values,
                               hovertemplate=f"{LABELS[xc]} %{{x}}<br>{LABELS[yc]} %{{y}}<br>CHD rate %{{z:.1f}}%<br>%{{customdata}} patients<extra></extra>",
                               colorbar=dict(title="CHD %", thickness=14)))
    fig.update_xaxes(title=LABELS[xc]); fig.update_yaxes(title=LABELS[yc])
    return style(fig, 460, False)


@st.cache_data
def missing_fig():
    df = load_data()
    m = (df.isna().mean() * 100)
    m = m[m > 0].sort_values()
    fig = go.Figure(go.Bar(x=m.values, y=[LABELS[i] for i in m.index], orientation="h", text=[f"{v:.1f}%" for v in m.values], textposition="outside",
                           marker=dict(color=m.values, colorscale="Sunsetdark", line=dict(width=0))))
    fig.update_xaxes(title="% of patients missing", range=[0, m.max() * 1.25])
    return style(fig, 520, False).update_layout(margin=dict(l=8, r=8, t=36, b=8))


def per_hundred(c):
    v = [c["tn"], c["fp"], c["fn"], c["tp"]]
    n = sum(v)
    raw = [x / n * 100 for x in v]
    fl = [int(np.floor(r)) for r in raw]
    for _, i in sorted(((r - f, i) for i, (r, f) in enumerate(zip(raw, fl))), reverse=True)[:100 - sum(fl)]:
        fl[i] += 1
    return fl


@st.cache_data
def picto_fig(name):
    per = per_hundred(load_json("metrics.json")["results"][name]["test"]["confusion"])
    seq = ["tn"] * per[0] + ["fp"] * per[1] + ["fn"] * per[2] + ["tp"] * per[3]
    lab = {"tn": "Healthy, not flagged", "fp": "Healthy, flagged (false alarm)", "fn": "CHD, missed", "tp": "CHD, caught"}
    fig = go.Figure()
    for k in ["tn", "fp", "fn", "tp"]:
        idx = [i for i, s in enumerate(seq) if s == k]
        fig.add_trace(go.Scatter(x=[i % 10 for i in idx], y=[9 - i // 10 for i in idx], mode="markers", name=f"{lab[k]} ({len(idx)})",
                                 marker=dict(size=20, color=DOT[k], line=dict(color="white", width=2)), hovertemplate=lab[k] + "<extra></extra>"))
    fig.update_xaxes(visible=False, range=[-.7, 9.7]).update_yaxes(visible=False, range=[-.7, 9.7], scaleanchor="x")
    return style(fig, 430).update_layout(title=f"100 patients screened with {name}")


@st.cache_data
def confusion_fig(name):
    c = load_json("metrics.json")["results"][name]["test"]["confusion"]
    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted: no CHD", "Predicted: CHD"], y=["Actual: no CHD", "Actual: CHD"], colorscale="Viridis",
                               text=[[f"{v}" for v in r] for r in z], texttemplate="<b>%{text}</b>", textfont=dict(size=26, color="white"), showscale=False, xgap=4, ygap=4))
    fig.update_yaxes(autorange="reversed")
    return style(fig, 300, False)


@st.cache_data
def meters_fig(name):
    t = load_json("metrics.json")["results"][name]["test"]
    items = [("Sensitivity", t["recall_sensitivity"], "#2ecc71"), ("Specificity", t["specificity"], "#4361ee"),
             ("Precision", t["precision"], "#fb8500"), ("F1 score", t["f1"], "#7209b7")]
    fig = go.Figure(go.Bar(x=[v * 100 for _, v, _ in items], y=[n for n, _, _ in items], orientation="h", marker_color=[c for *_, c in items],
                           text=[f"{v:.0%}" for _, v, _ in items], textposition="outside"))
    fig.update_xaxes(range=[0, 105], title="%").update_yaxes(autorange="reversed")
    return style(fig, 250, False)


@st.cache_data
def curves_fig():
    p = load_preds()
    y = p["y"].to_numpy()
    fig = make_subplots(rows=1, cols=3, subplot_titles=["ROC", "Precision-recall", "Calibration"], horizontal_spacing=0.08)
    for n in NAMES:
        s, col = p[slug(n)].to_numpy(), MODEL_COLORS[n]
        f, t, _ = roc_curve(y, s)
        fig.add_trace(go.Scatter(x=f, y=t, mode="lines", line=dict(color=col, width=3), name=n, legendgroup=n), row=1, col=1)
        pr, rc, _ = precision_recall_curve(y, s)
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=col, width=3), name=n, legendgroup=n, showlegend=False), row=1, col=2)
        fr, mp = calibration_curve(y, s, n_bins=8, strategy="quantile")
        fig.add_trace(go.Scatter(x=mp, y=fr, mode="lines+markers", line=dict(color=col, width=2.5), marker=dict(size=8), name=n, legendgroup=n, showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines", line=dict(color="gray", dash="dash"), showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="1 − specificity", row=1, col=1).update_yaxes(title_text="sensitivity", row=1, col=1)
    fig.update_xaxes(title_text="recall", row=1, col=2).update_yaxes(title_text="precision", row=1, col=2)
    fig.update_xaxes(title_text="predicted risk", row=1, col=3).update_yaxes(title_text="observed CHD rate", row=1, col=3)
    return style(fig, 420)


@st.cache_data
def auc_fig():
    R = load_json("metrics.json")["results"]
    fig = go.Figure()
    fig.add_bar(x=NAMES, y=[R[n]["cv"]["roc_auc"] for n in NAMES], name="5-fold CV", marker_color=[MODEL_COLORS[n] for n in NAMES],
                text=[f"{R[n]['cv']['roc_auc']:.3f}" for n in NAMES], textposition="outside")
    fig.add_bar(x=NAMES, y=[R[n]["test"]["roc_auc"] for n in NAMES], name="Held-out test", marker=dict(color=[MODEL_COLORS[n] for n in NAMES], pattern=dict(shape="/")),
                opacity=0.7, text=[f"{R[n]['test']['roc_auc']:.3f}" for n in NAMES], textposition="outside")
    fig.update_yaxes(range=[0.5, 0.8], title="ROC AUC").update_layout(barmode="group")
    return style(fig, 400).update_layout(title="Cross-validation (solid) vs test (hatched)")


@st.cache_data
def radar_fig():
    R = load_json("metrics.json")["results"]
    axes = ["Sensitivity", "Specificity", "Balanced accuracy", "Test AUC", "CV AUC"]
    fig = go.Figure()
    for n in NAMES:
        r = R[n]
        v = [r["test"]["recall_sensitivity"], r["test"]["specificity"], r["test"]["balanced_accuracy"], r["test"]["roc_auc"], r["cv"]["roc_auc"]]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], fill="toself", name=n, line=dict(color=MODEL_COLORS[n], width=3), opacity=0.55))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0.5, 0.75], tickformat=".2f")))
    return style(fig, 420).update_layout(title="Model profile (centre = 0.5, a coin flip)")


@st.cache_data
def importance_fig():
    d = pd.DataFrame(load_json("feature_importance.json")).sort_values("importance")
    fig = go.Figure(go.Bar(x=d.importance.clip(lower=0), y=d.label, orientation="h", error_x=dict(array=d["std"], color="rgba(0,0,0,.35)"),
                           marker=dict(color=d.importance.clip(lower=0), colorscale="Turbo", line=dict(width=0))))
    fig.update_xaxes(title="drop in test AUC when the feature is shuffled")
    return style(fig, 460, False).update_layout(title="What drives the model")


@st.cache_data
def ann_fig():
    h = load_json("ann_history.json")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss (dotted = chosen epoch)", "Validation AUC"])
    fig.add_trace(go.Scatter(y=h["train_loss"], name="train", line=dict(color="#4361ee", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(y=h["val_loss"], name="validation", line=dict(color="#f72585", width=3)), row=1, col=1)
    fig.add_vline(x=h["best_epoch"] - 1, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_trace(go.Scatter(y=h["val_auc"], name="val AUC", line=dict(color="#2a9d8f", width=3), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="epoch")
    return style(fig, 340)


def threshold_table(name):
    p = load_preds()
    y, s = p["y"].to_numpy(), p[slug(name)].to_numpy()
    ths = np.round(np.arange(0.02, 0.60, 0.005), 3)
    rows = []
    for t in ths:
        pred = s >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        rows.append((t, tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1), pred.mean()))
    return pd.DataFrame(rows, columns=["t", "sens", "spec", "prec", "flag"])


def threshold_fig(name, t):
    d = threshold_table(name)
    fig = go.Figure()
    for col, lab, c in [("sens", "Sensitivity", "#2ecc71"), ("spec", "Specificity", "#4361ee"), ("prec", "Precision", "#fb8500"), ("flag", "Share flagged", "#7209b7")]:
        fig.add_trace(go.Scatter(x=d.t * 100, y=d[col] * 100, mode="lines", name=lab, line=dict(color=c, width=3.5)))
    fig.add_vline(x=t * 100, line_color="#e5383b", line_width=3)
    fig.update_xaxes(title="decision threshold (% risk)").update_yaxes(title="%", range=[0, 102])
    return style(fig, 380)


# ====================================================================== page: risk calculator
PRESETS = {
    "low": dict(male=0, age=35, sysBP=112, diaBP=72, currentSmoker=False, cigsPerDay=None, prevalentHyp=False, diabetes=False, prevalentStroke=False,
                BPMeds=False, totChol=180, glucose=75, BMI=22.0, heartRate=68, education=None),
    "mid": dict(male=1, age=50, sysBP=130, diaBP=84, currentSmoker=False, cigsPerDay=None, prevalentHyp=False, diabetes=False, prevalentStroke=False,
                BPMeds=False, totChol=235, glucose=80, BMI=26.0, heartRate=75, education=None),
    "high": dict(male=1, age=62, sysBP=165, diaBP=98, currentSmoker=True, cigsPerDay=30, prevalentHyp=True, diabetes=True, prevalentStroke=False,
                 BPMeds=True, totChol=280, glucose=180, BMI=31.0, heartRate=82, education=1),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def read_patient():
    s = st.session_state
    return {"male": int(s.male), "age": s.age, "sysBP": s.sysBP, "diaBP": s.diaBP, "currentSmoker": int(s.currentSmoker),
            "cigsPerDay": (s.cigsPerDay if s.currentSmoker else None), "prevalentHyp": int(s.prevalentHyp), "diabetes": int(s.diabetes),
            "prevalentStroke": int(s.prevalentStroke), "BPMeds": int(s.BPMeds), "totChol": s.totChol, "glucose": s.glucose, "BMI": s.BMI,
            "heartRate": s.heartRate, "education": s.education}


@st.fragment
def whatif_section(model_name, patient, p):
    model = get_model(model_name)
    c1, c2 = st.columns([7, 5], gap="large")
    with c1:
        st.subheader("How risk changes across a range")
        st.caption("Everything else stays as entered. Press the replay button to watch the curve draw.")
        label = st.pills("Factor to vary", list(SWEEPS), default="Age", key="sweep_pick") or "Age"
        xs, ys = sweep_data(model, patient, label)
        key, _, _, _, unit = SWEEPS[label]
        cur_x = patient.get(key) if key != "cigsPerDay" else (patient["cigsPerDay"] or 0)
        st.plotly_chart(sweep_fig(xs, ys, label, None if cur_x is None else (cur_x, p * 100), unit), theme=None, key="sweep_chart")
        explain("sweep_chart", st)
    with c2:
        st.subheader("What if one thing changed?")
        st.caption("Model estimates with everything else kept the same. Statistical associations, not a treatment plan.")
        rows = scenarios_data(model, patient, p)
        if rows:
            st.plotly_chart(scenarios_fig(rows, p), theme=None, key="scen_chart")
            explain("scen_chart", st)
        else:
            st.info("Nothing in this profile sits above the usual reference range, so there is no single change to model. Try the high-risk example.")


def page_risk(model_name):
    st.markdown("""
    <div class="hero"><div style="position:relative;z-index:1">
      <div class="eyebrow">Framingham Heart Study cohort</div>
      <h1>How likely is heart disease in the next ten years?</h1>
      <p>Enter a patient profile and watch the estimate respond. Then explore how the risk changes with each factor.</p>
      <div class="chips"><span class="chip"><b>4,240</b> patients</span><span class="chip"><b>15</b> risk factors</span><span class="chip"><b>4</b> models compared</span></div></div>
      <svg viewBox="0 0 520 140" preserveAspectRatio="none"><path class="ecg-b" pathLength="1" d="M0 80H70l14-6 14 6h40l10 8 12-62 14 98 12-44 10 8h60l14-10 14 10h50l10 8 12-62 14 98 12-44 10 8h80"/>
      <path class="ecg-l" pathLength="1" d="M0 80H70l14-6 14 6h40l10 8 12-62 14 98 12-44 10 8h80"/></svg></div>""", unsafe_allow_html=True)

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Patient profile")
        b1, b2, b3 = st.columns(3)
        b1.button("🟢 Low-risk", on_click=apply_preset, args=("low",), width="stretch")
        b2.button("🟡 Average", on_click=apply_preset, args=("mid",), width="stretch")
        b3.button("🔴 High-risk", on_click=apply_preset, args=("high",), width="stretch")
        a, b = st.columns(2)
        a.number_input("Age (years)", 18, 100, key="age")
        b.radio("Sex", [0, 1], format_func=lambda v: "Male" if v else "Female", horizontal=True, key="male")
        a, b = st.columns(2)
        a.number_input("Systolic BP (mmHg)", 70, 300, key="sysBP")
        b.number_input("Diastolic BP (mmHg)", 40, 160, key="diaBP")
        t1, t2 = st.columns(2)
        t1.toggle("Current smoker", key="currentSmoker")
        t2.toggle("Hypertension", key="prevalentHyp")
        t1.toggle("Diabetes", key="diabetes")
        t2.toggle("Previous stroke", key="prevalentStroke")
        t1.toggle("On BP medication", key="BPMeds")
        if st.session_state.currentSmoker:
            st.number_input("Cigarettes per day", 0, 100, value=None, placeholder="e.g. 15", key="cigsPerDay")
        st.markdown("**Measurements** <small>optional, estimated from the cohort if left blank</small>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.number_input("Total cholesterol (mg/dL)", 80, 700, value=None, placeholder="e.g. 230", key="totChol")
        b.number_input("Glucose (mg/dL)", 30, 500, value=None, placeholder="e.g. 80", key="glucose")
        a, b = st.columns(2)
        a.number_input("BMI", 10.0, 70.0, value=None, step=0.1, placeholder="e.g. 25.5", key="BMI")
        b.number_input("Resting heart rate (bpm)", 30, 200, value=None, placeholder="e.g. 75", key="heartRate")
        st.selectbox("Education", [None, 1, 2, 3, 4], format_func=lambda v: "Not specified" if v is None else EDU[v], key="education")

    patient = read_patient()
    with right:
        if patient["diaBP"] >= patient["sysBP"]:
            st.error("Diastolic BP must be lower than systolic BP.")
            return None
        model = get_model(model_name)
        frame = pd.DataFrame([patient])
        p = float(model.predict_proba(frame)[0, 1])
        thr = CARD["thresholds"][model_name]
        cat = cat_of(p)
        st.plotly_chart(gauge_fig(p), theme=None, key="gauge")
        explain("gauge", st)
        ratio = p / BASE
        vs = (f"About <b>{ratio:.1f}×</b> the cohort average of {BASE:.1%}" if ratio >= 1.15 else
              f"About <b>{(1 - ratio) * 100:.0f}% below</b> the cohort average of {BASE:.1%}" if ratio <= 0.85 else f"Close to the cohort average of {BASE:.1%}")
        bg = {"low": "#d8f3e3", "moderate": "#fdf0c8", "high": "#fbd9d6"}[cat]
        fg = {"low": "#1e7a45", "moderate": "#8a5a00", "high": "#a31f1a"}[cat]
        st.markdown(f"<div style='text-align:center'><span class='pill pill-{cat}'>{cat.capitalize()} risk</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;margin-top:8px'>{vs}</div>", unsafe_allow_html=True)
        flagged = p >= thr
        (st.warning if flagged else st.success)(
            f"{'Above' if flagged else 'Below'} the model's screening threshold ({thr:.1%}): this profile would be "
            f"**{'flagged for follow-up' if flagged else 'not flagged'}**.")
        items = local_factors(model, frame)
        if items:
            st.plotly_chart(factors_fig(items), theme=None, key="factors")
            explain("factors", st)
        missing = [k for k in ["totChol", "glucose", "BMI", "heartRate"] if patient[k] is None]
        st.caption((f"{len(missing)} measurement(s) left blank and estimated from the cohort median. " if missing else "") +
                   f"Model: {model_name}. These are statistical associations in one cohort, not causes.")

    st.divider()
    cmp_cols = st.columns([5, 7], gap="large")
    with cmp_cols[0]:
        st.plotly_chart(models_fig({n: float(get_model(n).predict_proba(frame)[0, 1]) for n in NAMES}), theme=None, key="cmp")
        explain("cmp", st)
    with cmp_cols[1]:
        st.markdown("#### Reading the result")
        st.markdown(f"- **Low** is under {BANDS['low_below']:.0%}, **moderate** {BANDS['low_below']:.0%}–{BANDS['high_from']:.0%}, **high** {BANDS['high_from']:.0%} or more. The cohort average is {BASE:.1%}.\n"
                    f"- The blue line on the gauge marks the cohort average.\n"
                    f"- The four models usually agree; large disagreement means the profile is unusual.\n"
                    f"- Discrimination is moderate (AUC about 0.70–0.73), so treat this as a screening aid.")
    st.divider()
    whatif_section(model_name, patient, p)
    return p


# ====================================================================== page: explorer
@st.fragment
def page_explore():
    df = load_data()
    k = st.columns(4)
    k[0].metric("Patients", f"{len(df):,}")
    k[1].metric("Developed CHD in 10 years", f"{df[TARGET].mean():.1%}")
    k[2].metric("Risk factors per patient", f"{len(FEATURES)}")
    k[3].metric("Patients with a missing value", f"{df.isna().any(axis=1).mean():.0%}")
    c1, c2 = st.columns([4, 6], gap="large")
    c1.subheader("The outcome")
    c1.plotly_chart(donut_fig(), theme=None, key="donut")
    explain("donut", c1)
    c2.subheader("Risk multipliers")
    c2.plotly_chart(multipliers_fig(), theme=None, key="mult")
    explain("mult", c2)
    st.subheader("CHD rate by risk factor")
    pick = st.pills("Pick a factor", RATE_LABELS, default="Age", key="rate_pick") or "Age"
    st.plotly_chart(rate_fig(pick), theme=None, key="rate")
    explain("rate", st)
    st.subheader("How the factors relate to each other")
    st.plotly_chart(heat_fig(), theme=None, key="heat")
    explain("heat", st)
    st.subheader("Distribution explorer")
    a, b = st.columns([1, 3])
    var = a.selectbox("Variable", CONTINUOUS, format_func=lambda c: LABELS[c], key="dist_var")
    b.plotly_chart(hist_fig(var), theme=None, key="hist")
    explain("hist", b)
    st.subheader("Where is risk concentrated?")
    a, b, c = st.columns([1, 1, 3])
    xc = a.selectbox("Across", CONTINUOUS, index=0, format_func=lambda c: LABELS[c], key="rm_x")
    yc = b.selectbox("Down", CONTINUOUS, index=3, format_func=lambda c: LABELS[c], key="rm_y")
    if xc == yc:
        c.info("Pick two different variables.")
    else:
        c.plotly_chart(rate_map_fig(xc, yc), theme=None, key="ratemap")
        explain("ratemap", c)
    c1, c2 = st.columns([3, 2], gap="large")
    c1.subheader("Spread of every measurement")
    c2.subheader("Missing data")
    c1.plotly_chart(violins_fig(), theme=None, key="violins")
    explain("violins", c1)
    c2.plotly_chart(missing_fig(), theme=None, key="missing")
    explain("missing", c2)



# ====================================================================== glossary & final-results content
FEATURE_GLOSSARY = [
    ("male", "Sex", "1 = male, 0 = female"), ("age", "Age", "Age in years at the first exam"),
    ("education", "Education level", "1 = some high school ... 4 = college or above"),
    ("currentSmoker", "Current smoker", "1 = smokes now"), ("cigsPerDay", "Cigarettes per day", "Average cigarettes smoked daily"),
    ("BPMeds", "Blood-pressure medication", "1 = on BP medication"), ("prevalentStroke", "Prior stroke", "1 = has had a stroke"),
    ("prevalentHyp", "Hypertension", "1 = diagnosed with high blood pressure"), ("diabetes", "Diabetes", "1 = diabetic"),
    ("totChol", "Total cholesterol", "mg/dL"), ("sysBP", "Systolic blood pressure", "Pressure when the heart beats, mmHg"),
    ("diaBP", "Diastolic blood pressure", "Pressure between beats, mmHg"), ("BMI", "Body mass index", "Weight (kg) / height (m) squared"),
    ("heartRate", "Resting heart rate", "Beats per minute"), ("glucose", "Blood glucose", "mg/dL"),
    ("TenYearCHD", "Outcome (target)", "1 = coronary heart disease within 10 years"),
]
TERM_GLOSSARY = [
    ("CHD", "Coronary heart disease", "Narrowing of the arteries that feed the heart muscle; the outcome we predict."),
    ("ANN / MLP", "Artificial neural network / multilayer perceptron", "A small network of layers of weighted units that learns non-linear patterns."),
    ("XGBoost", "Extreme gradient boosting", "Many small decision trees built one after another, each fixing the last one's errors."),
    ("RF", "Random forest", "Many decision trees trained on random subsets whose votes are averaged."),
    ("LR", "Logistic regression", "A linear model that turns a weighted sum of factors into a probability."),
    ("CV", "Cross-validation", "Splitting the training data into folds, training on some and scoring on the rest, to estimate performance fairly."),
    ("AUC / ROC-AUC", "Area under the ROC curve", "Chance that a random CHD patient gets a higher risk than a random healthy one. 0.5 = coin flip, 1.0 = perfect."),
    ("PR-AUC", "Area under the precision-recall curve", "Like AUC but focused on the rare positive class; useful when CHD is uncommon."),
    ("Sensitivity / Recall", "True-positive rate", "Share of people who develop CHD that the model flags."),
    ("Specificity", "True-negative rate", "Share of healthy people the model correctly leaves unflagged."),
    ("Precision", "Positive predictive value", "Of the people flagged, the share who really develop CHD."),
    ("F1", "F1 score", "Harmonic mean of precision and recall."),
    ("Youden's J", "Sensitivity + specificity - 1", "Used to pick the threshold that best balances catching cases against false alarms."),
    ("Threshold", "Decision cut-off", "Predicted risk above which a person is flagged for follow-up."),
    ("TP / FP / TN / FN", "Confusion-matrix cells", "True/false positives and negatives: correct flags, false alarms, correct all-clears, missed cases."),
    ("Brier score", "Probability error", "Mean squared gap between predicted risk and outcome. Lower is better."),
    ("Calibration", "Probability honesty", "Whether 20% predicted risk really means about 20 in 100 develop CHD."),
    ("Data leakage", "Test information leaking into training", "Avoided by learning medians and scaling from training data only."),
    ("Imputation", "Filling missing values", "Median for numbers, mode for categories, plus a missing-glucose flag."),
]
HOW_STEPS = [
    ("1. Collect", "Load the Framingham Heart Study extract (about 4,240 patients, 15 risk factors, outcome `TenYearCHD`) and audit missing values and duplicates."),
    ("2. Split", "Hold out 20% of patients as a locked test set, keeping the same CHD rate in both parts (stratified)."),
    ("3. Clean", "Impute gaps and standardise numeric columns. Everything is learned from the training split only, and refit inside every CV fold, so nothing leaks."),
    ("4. Train", "Fit four pipelines: logistic regression, random forest, XGBoost and a PyTorch ANN. No class weighting, so probabilities stay interpretable."),
    ("5. Validate", "5-fold cross-validation gives out-of-fold predictions. The threshold maximises Youden's J on them, and the served model is chosen by CV AUC alone."),
    ("6. Test", "Each model is scored once on the untouched test set, which is the honest final check."),
    ("7. Explain", "For one patient each factor is swapped to a typical value to show what pushes risk up or down."),
    ("8. Serve", "Portable artifacts (weights and preprocessing parameters as JSON/CSV) are read by this dashboard, which needs no custom classes."),
]


def final_verdict(R, serving):
    """Plain-language comparison built only from the metrics in metrics.json (so it always matches the data)."""
    names = list(R)
    top = lambda key, part: max(names, key=lambda n: R[n][part][key])
    cv_best, auc_best = top("roc_auc", "cv"), top("roc_auc", "test")
    sens_best, spec_best = top("recall_sensitivity", "test"), top("specificity", "test")
    f1_best, brier_best = top("f1", "test"), min(names, key=lambda n: R[n]["test"]["brier"])
    spread = max(R[n]["cv"]["roc_auc"] for n in names) - min(R[n]["cv"]["roc_auc"] for n in names)
    rows = [("Best cross-validated AUC (model selection)", cv_best, R[cv_best]["cv"]["roc_auc"]),
            ("Best test AUC", auc_best, R[auc_best]["test"]["roc_auc"]),
            ("Highest sensitivity (catches most CHD)", sens_best, R[sens_best]["test"]["recall_sensitivity"]),
            ("Highest specificity (fewest false alarms)", spec_best, R[spec_best]["test"]["specificity"]),
            ("Best F1", f1_best, R[f1_best]["test"]["f1"]),
            ("Lowest Brier score (best probabilities)", brier_best, R[brier_best]["test"]["brier"])]
    text = (f"**{serving}** is served: it has the best cross-validated AUC ({R[serving]['cv']['roc_auc']:.3f}) and was picked before the test set was opened. "
            f"The {len(names)} models sit within {spread:.3f} AUC of each other in cross-validation, so the gaps are small relative to the test set's noise of about ±0.03. "
            f"{'No model wins every metric, and a simpler, more explainable model is preferred when accuracy is this close.' if len({r[1] for r in rows}) > 1 else 'It also leads on every metric above.'}")
    return rows, text


# ====================================================================== page: models
@st.fragment
def page_models(serving):
    R = load_json("metrics.json")["results"]
    st.caption(f"All four models land within about 0.02 AUC of each other in cross-validation, so the simplest, **{serving}**, is served by default. "
               f"Models were chosen on cross-validation only; the {CARD['n_test']}-patient test set is used for the final report (expect about ±0.03 of noise).")
    cols = st.columns(4)
    for col, n in zip(cols, NAMES):
        r = R[n]
        badge = '<span class="badge">served</span>' if n == serving else ""
        col.markdown(f"""<div class="mcard" style="--c:{MODEL_COLORS[n]}"><b>{n}</b>{badge}<div class="auc">{r['test']['roc_auc']:.3f}</div>
            <small>test AUC · CV {r['cv']['roc_auc']:.3f}<br>sensitivity {r['test']['recall_sensitivity']:.0%} · specificity {r['test']['specificity']:.0%}</small></div>""",
                     unsafe_allow_html=True)
    st.write("")
    sel = st.segmented_control("Inspect a model", NAMES, default=serving, key="inspect") or serving
    a, b = st.columns(2, gap="large")
    a.plotly_chart(picto_fig(sel), theme=None, key="picto")
    explain("picto", a)
    per = per_hundred(R[sel]["test"]["confusion"])
    a.caption(f"About **{per[2] + per[3]}** of these 100 develop CHD. The model catches **{per[3]}** and misses **{per[2]}**. "
              f"Of the **{per[0] + per[1]}** who stay healthy it flags **{per[1]}** anyway.")
    b.markdown("##### Confusion matrix (test set)")
    b.plotly_chart(confusion_fig(sel), theme=None, key="conf")
    explain("conf", b)
    b.markdown("##### Detection quality")
    b.plotly_chart(meters_fig(sel), theme=None, key="meters")
    explain("meters", b)

    st.subheader("Threshold explorer")
    st.caption("Drag the threshold to see the trade-off. A lower threshold catches more CHD cases but raises more false alarms.")
    t = st.slider("Decision threshold (predicted risk)", 0.02, 0.59, float(CARD["thresholds"][sel]), 0.005, format="%.3f", key="thr")
    d = threshold_table(sel)
    row = d.iloc[(d.t - t).abs().argmin()]
    m = st.columns(4)
    m[0].metric("Sensitivity", f"{row.sens:.0%}")
    m[1].metric("Specificity", f"{row.spec:.0%}")
    m[2].metric("Precision", f"{row.prec:.0%}")
    m[3].metric("Patients flagged", f"{row.flag:.0%}")
    st.plotly_chart(threshold_fig(sel, t), theme=None, key="thrfig")
    explain("thrfig", st)

    st.subheader("Model comparison")
    st.plotly_chart(curves_fig(), theme=None, key="curves")
    explain("curves", st)
    a, b = st.columns(2, gap="large")
    a.plotly_chart(auc_fig(), theme=None, key="auc")
    explain("auc", a)
    b.plotly_chart(radar_fig(), theme=None, key="radar")
    explain("radar", b)
    st.subheader("Full metrics")
    table = pd.DataFrame([{"Model": n, "CV AUC": R[n]["cv"]["roc_auc"], "Test AUC": R[n]["test"]["roc_auc"], "Test PR-AUC": R[n]["test"]["pr_auc"],
                           "Sensitivity": R[n]["test"]["recall_sensitivity"], "Specificity": R[n]["test"]["specificity"],
                           "Precision": R[n]["test"]["precision"], "F1": R[n]["test"]["f1"], "Threshold": R[n]["cv"]["threshold"]} for n in NAMES]).set_index("Model")
    st.dataframe(table.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.3f}"), width="stretch")
    st.subheader("Final results: which model wins?")
    rows, verdict = final_verdict(R, serving)
    st.success(verdict)
    st.dataframe(pd.DataFrame([{"Criterion": c, "Winner": w, "Value": v} for c, w, v in rows]).set_index("Criterion").style.format({"Value": "{:.3f}"}), width="stretch")
    fin = pd.DataFrame([{"Model": n, "Rank (CV AUC)": 0, "CV AUC": R[n]["cv"]["roc_auc"], "Test AUC": R[n]["test"]["roc_auc"],
                         "Sensitivity": R[n]["test"]["recall_sensitivity"], "Specificity": R[n]["test"]["specificity"],
                         "Brier": R[n]["test"]["brier"]} for n in NAMES])
    fin["Rank (CV AUC)"] = fin["CV AUC"].rank(ascending=False).astype(int)
    st.dataframe(fin.sort_values("Rank (CV AUC)").set_index("Model").style.format({c: "{:.3f}" for c in fin.columns[2:]}), width="stretch")
    a, b = st.columns(2, gap="large")
    a.plotly_chart(importance_fig(), theme=None, key="imp")
    explain("imp", a)
    b.plotly_chart(ann_fig(), theme=None, key="ann")
    explain("ann", b)


# ====================================================================== page: about
def page_about():
    st.markdown("### How CardioScope works")
    st.caption("From raw cohort data to a prediction in six steps. Every step only uses information it is allowed to see.")
    steps = [("Collect", "4,240 patients, 15 risk factors, one 10-year outcome", "#4361ee"), ("Split", "80 / 20 stratified. The test set stays locked away", "#7209b7"),
             ("Clean", "Impute and scale using training data only", "#f72585"), ("Train", "Logistic regression, random forest, XGBoost and an ANN", "#fb8500"),
             ("Validate", "5-fold cross-validation picks the model and threshold", "#2a9d8f"), ("Serve", "This dashboard, with per-patient explanations", "#e5383b")]
    cols = st.columns(6)
    for i, (col, (t, d, c)) in enumerate(zip(cols, steps)):
        col.markdown(f"<div class='step' style='background:linear-gradient(145deg,{c},{c}cc);animation-delay:{i * 0.12}s'><div class='n'>{i + 1}</div><b>{t}</b><span>{d}</span></div>", unsafe_allow_html=True)
    st.write("")
    with st.expander("Step-by-step: how it works in detail"):
        for t, d in HOW_STEPS:
            st.markdown(f"**{t}** {d}")
    g1, g2 = st.tabs(["📖 Abbreviations & terms", "🧾 Dataset columns"])
    with g1:
        q = st.text_input("Search terms", placeholder="e.g. AUC, sensitivity, CV", key="gloss_q")
        gt = pd.DataFrame(TERM_GLOSSARY, columns=["Abbreviation", "Stands for", "Meaning"])
        if q:
            gt = gt[gt.apply(lambda r: q.lower() in " ".join(r).lower(), axis=1)]
        st.dataframe(gt.set_index("Abbreviation"), width="stretch")
    with g2:
        st.dataframe(pd.DataFrame(FEATURE_GLOSSARY, columns=["Column", "Meaning", "Detail"]).set_index("Column"), width="stretch")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Questions")
        with st.expander("Why is a simple model served and not the neural network?", expanded=True):
            st.write("All four models land within about 0.02 AUC of each other in cross-validation. Logistic regression scored highest, and on a small table like this a linear model is as accurate as a neural network while being much easier to explain.")
        with st.expander("What do low, moderate and high mean?"):
            st.write(f"Low is below {BANDS['low_below']:.0%}, moderate is {BANDS['low_below']:.0%} to {BANDS['high_from']:.0%}, high is {BANDS['high_from']:.0%} or more. The cohort average is about {BASE:.0%}. Separately, the screening threshold decides which profiles the model would flag for follow-up.")
        with st.expander("Why are test scores lower than cross-validation?"):
            st.write("The test set has about 850 patients, around 130 with CHD, so its AUC carries roughly ±0.03 of noise. Models were chosen on cross-validation only, so the test result is an honest, untouched check.")
        with st.expander("Is this medical advice?"):
            st.write("No. It is an educational project on a historical cohort. It cannot diagnose anyone and should never replace a clinician.")
    with b:
        st.markdown("#### Limitations")
        st.markdown("- The cohort began in Massachusetts in 1948 and may not represent other populations or modern treatment.\n"
                    "- Probabilities are not calibrated for individuals, and hyperparameters were not tuned.\n"
                    "- Moderate discrimination means many people flagged high risk will not develop CHD.\n"
                    f"- {CARD['disclaimer']}")
        st.markdown("#### Reproduce")
        st.code("1. Run all cells of CardioScope_EndToEnd.ipynb  (trains, evaluates, writes artifacts/)\n2. streamlit run streamlit_app.py", language="text")
        v = CARD["versions"]
        st.caption(f"Trained {CARD['trained_at'][:10]} · scikit-learn {v['scikit-learn']} · xgboost {v['xgboost']} · torch {v['torch']}")


# ====================================================================== main
if "age" not in st.session_state:
    apply_preset("mid")

with st.sidebar:
    st.markdown("## 🫀 CardioScope")
    st.toggle("🌙 Dark mode", key="dark")
    st.markdown("<span class='live'></span>Model ready · UI v3", unsafe_allow_html=True)
    model_name = st.selectbox("Serving model", NAMES, index=NAMES.index(CARD["best_model"]), help="Which model produces the risk estimate on the calculator tab.")
    st.caption(f"Screening threshold for this model: **{CARD['thresholds'][model_name]:.1%}**")
    st.divider()
    st.caption(CARD["disclaimer"])

t1, t2, t3, t4 = st.tabs(["🧮 Risk calculator", "📊 Data explorer", "🏆 Model performance", "ℹ️ About"])
with t1:
    page_risk(model_name)
with t2:
    page_explore()
with t3:
    page_models(CARD["best_model"])
with t4:
    page_about()