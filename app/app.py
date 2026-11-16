# ==============================================================================
# Multi-Drug Interaction Prediction System
# Doctor portal (own use + patient prescriptions) and Patient portal
# ==============================================================================
import streamlit as st
import sys, os, tempfile, sqlite3, hashlib, secrets
import plotly.graph_objects as go
from datetime import datetime
from xml.sax.saxutils import escape

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.resolver import DrugResolver
from src.inference import DDIPredictor
from src.report_generator import ClinicalReportGenerator

st.set_page_config(page_title="Multi-Drug Interaction Prediction", page_icon=None,
                   layout="wide", initial_sidebar_state="collapsed")

# ==============================================================================
# DESIGN SYSTEM
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif !important;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0;}
.stApp {background:linear-gradient(135deg,#EEF2FF 0%,#FDF2F8 55%,#ECFDF5 100%); color:#0F172A;}
.block-container {max-width:1250px !important; padding-top:1rem !important; padding-bottom:3rem !important;}

/* top nav */
.nav {background:rgba(255,255,255,.88); backdrop-filter:blur(10px); border:1px solid #E2E8F0;
  border-radius:16px; padding:12px 20px; display:flex; justify-content:space-between; align-items:center;
  box-shadow:0 4px 18px rgba(79,70,229,.08); margin-bottom:18px;}
.nav-title {font-weight:800; font-size:1.1rem;}
.pill {background:#F1F5F9; border:1px solid #E2E8F0; padding:6px 14px; border-radius:20px; font-weight:600; font-size:.85rem;}

/* banners */
.banner {border-radius:20px; padding:26px 32px; color:#fff; margin-bottom:18px; display:flex;
  justify-content:space-between; align-items:center; box-shadow:0 14px 30px -10px rgba(0,0,0,.25);}
.banner-doc {background:linear-gradient(135deg,#065F46,#059669 60%,#0891B2);}
.banner-pat {background:linear-gradient(135deg,#4338CA,#7C3AED 60%,#DB2777);}
.banner h2 {margin:0 0 4px; font-size:1.8rem; font-weight:800; color:#fff;}
.banner p {margin:0; opacity:.92;}
.avatar {width:70px; height:70px; border-radius:50%; background:rgba(255,255,255,.2); display:flex;
  align-items:center; justify-content:center; font-size:1.8rem; font-weight:800;}

/* kpi */
.kpi {background:#fff; border:1px solid #E2E8F0; border-radius:18px; padding:18px 22px;
  box-shadow:0 6px 16px rgba(15,23,42,.05);}
.k-blue {border-left:6px solid #4F46E5;} .k-amber {border-left:6px solid #EA580C;}
.k-purple {border-left:6px solid #9333EA;}
.kpi-l {font-size:.8rem; color:#64748B; font-weight:600;}
.kpi-n {font-size:2rem; font-weight:800; line-height:1.2;}
.kpi-s {font-size:.78rem; color:#64748B;}

/* cards */
.card-head {border-radius:16px 16px 0 0; padding:14px 20px; color:#fff; font-weight:800; font-size:1.05rem;
  display:flex; justify-content:space-between; align-items:center; margin-top:10px;}
.h-green {background:linear-gradient(90deg,#059669,#0D9488);} .h-blue {background:linear-gradient(90deg,#2563EB,#4F46E5);}
.h-purple {background:linear-gradient(90deg,#7C3AED,#DB2777);}
.card-body {background:#fff; border:1px solid #E2E8F0; border-top:none; border-radius:0 0 16px 16px; padding:6px 18px 16px;
  box-shadow:0 6px 16px rgba(15,23,42,.05); margin-bottom:14px;}
.rx-row {display:flex; justify-content:space-between; align-items:center; background:#F8FAFC; border:1px solid #E2E8F0;
  border-radius:12px; padding:10px 14px; margin:8px 0;}
.badge {font-size:.72rem; font-weight:800; padding:3px 10px; border-radius:20px; display:inline-block; margin:2px 4px 2px 0;}
.b-red {background:#FEE2E2; color:#991B1B;} .b-orange {background:#FFEDD5; color:#9A3412;}
.b-amber {background:#FEF3C7; color:#92400E;} .b-green {background:#DCFCE7; color:#166534;}
.b-blue {background:#DBEAFE; color:#1E40AF;}
.ddi {border-radius:0 14px 14px 0; padding:14px 18px; margin:10px 0; background:#fff; border:1px solid #E2E8F0;
  box-shadow:0 4px 12px rgba(15,23,42,.04);}
.ddi-red {border-left:6px solid #DC2626; background:#FEF2F2;} .ddi-orange {border-left:6px solid #EA580C; background:#FFF7ED;}
.ddi-amber {border-left:6px solid #D97706; background:#FFFBEB;} .ddi-green {border-left:6px solid #16A34A; background:#F0FDF4;}
.ddi-title {font-size:1.1rem; font-weight:800; margin:6px 0;}
.se-label {font-size:.8rem; font-weight:700; color:#475569; margin-top:8px;}

/* inputs / buttons / tabs */
.stButton>button, .stDownloadButton>button {border-radius:12px; font-weight:700; border:1px solid #CBD5E1;}
.stButton>button[kind="primary"], .stDownloadButton>button {background:linear-gradient(90deg,#4F46E5,#9333EA); color:#fff; border:none;
  box-shadow:0 8px 18px -6px rgba(79,70,229,.5);}
.stButton>button:hover {transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"] {background:rgba(255,255,255,.85); border-radius:14px; padding:5px; gap:6px; border:1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"] {border-radius:10px; padding:8px 18px; font-weight:700;}
.stTabs [aria-selected="true"] {background:linear-gradient(90deg,#4F46E5,#9333EA); color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE (SQLite)
# ==============================================================================
DB_PATH = os.path.join(ROOT_DIR, "data", "pharmaguard_users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def hash_pw(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120000).hex()

def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL, salt TEXT NOT NULL, role TEXT NOT NULL, created TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS prescriptions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, doctor_id INTEGER NOT NULL,
            drug TEXT NOT NULL, note TEXT, created TEXT NOT NULL)""")
        # remove the old built-in demo accounts (and their prescriptions) if they exist in an older database
        demo = [r["id"] for r in con.execute(
            "SELECT id FROM users WHERE email IN ('doctor@jnnce.edu','patient@health.org')")]
        for uid in demo:
            con.execute("DELETE FROM prescriptions WHERE patient_id=? OR doctor_id=?", (uid, uid))
            con.execute("DELETE FROM users WHERE id=?", (uid,))

def get_user_by_email(email):
    with db() as con:
        return con.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()

def create_user(name, email, password, role):
    salt = secrets.token_hex(16)
    try:
        with db() as con:
            con.execute("INSERT INTO users(name,email,pw_hash,salt,role,created) VALUES(?,?,?,?,?,?)",
                        (name.strip(), email.strip().lower(), hash_pw(password, salt), salt, role,
                         datetime.now().strftime("%Y-%m-%d %H:%M")))
        return True
    except sqlite3.IntegrityError:
        return False

def verify_login(email, password, role):
    u = get_user_by_email(email)
    if u and u["role"] == role and hash_pw(password, u["salt"]) == u["pw_hash"]:
        return dict(u)
    return None

def list_patients():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT id,name,email FROM users WHERE role='patient' ORDER BY name")]

def add_rx(patient_id, doctor_id, drug, note):
    with db() as con:
        con.execute("INSERT INTO prescriptions(patient_id,doctor_id,drug,note,created) VALUES(?,?,?,?,?)",
                    (patient_id, doctor_id, drug, note, datetime.now().strftime("%Y-%m-%d %H:%M")))

def remove_rx(rx_id, doctor_id):
    with db() as con:   # a doctor can only remove their own prescriptions
        con.execute("DELETE FROM prescriptions WHERE id=? AND doctor_id=?", (rx_id, doctor_id))

def get_rx(patient_id):
    with db() as con:
        return [dict(r) for r in con.execute(
            """SELECT p.*, u.name AS doctor_name FROM prescriptions p JOIN users u ON u.id=p.doctor_id
               WHERE p.patient_id=? ORDER BY p.created DESC""", (patient_id,))]

init_db()

# ==============================================================================
# ENGINES
# ==============================================================================
@st.cache_resource
def load_clinical_engines():
    resolver = DrugResolver()
    predictor = DDIPredictor(checkpoint_path=os.path.join(ROOT_DIR, 'data', 'model_checkpoint.pt'))
    return resolver, predictor, ClinicalReportGenerator()

resolver, predictor, report_gen = load_clinical_engines()

# ==============================================================================
# SESSION STATE
# ==============================================================================
S = st.session_state
S.setdefault("user", None)
S.setdefault("own_drugs", [])      # doctor's own-use screener list (generic names)
S.setdefault("add_ctr", 0)
S.setdefault("auth_mode", "login")
S.setdefault("flash", "")

def logout():
    S.user = None
    S.own_drugs = []
    st.rerun()

def resolve_list(names):
    out = []
    for n in names:
        r = resolver.resolve(n)
        if r.get("resolved"):
            out.append(r)
    return out

# ==============================================================================
# SHARED UI HELPERS
# ==============================================================================
def kpi(col, label, num, sub, cls):
    col.markdown(f"""<div class="kpi {cls}"><div class="kpi-l">{label}</div><div class="kpi-n">{num}</div>
    <div class="kpi-s">{sub}</div></div>""", unsafe_allow_html=True)

def head(title, cls="h-blue", right=""):
    st.markdown(f'<div class="card-head {cls}"><span>{title}</span><span style="font-size:.8rem;opacity:.9">{right}</span></div>'
                f'<div class="card-body">', unsafe_allow_html=True)

def end():
    st.markdown("</div>", unsafe_allow_html=True)

def severity_style(pdata):
    sev = pdata.get("severity", "Moderate")
    if sev == "Major" and pdata.get("risk_score", 0.6) > 0.82:
        return "ddi-red", "b-red", "DON'T USE TOGETHER", "#DC2626"
    if sev == "Major":
        return "ddi-orange", "b-orange", "SERIOUS", "#EA580C"
    if sev == "Moderate":
        return "ddi-amber", "b-amber", "MONITOR CLOSELY", "#D97706"
    return "ddi-green", "b-green", "LOW RISK", "#16A34A"

def risky_pairs(pairwise):
    """Only the combinations that actually carry a risk (Major / Moderate)."""
    return {k: v for k, v in pairwise.items() if v.get("severity", "Moderate") in ("Major", "Moderate")}

RANK = {"Minor": 0, "Moderate": 1, "Major": 2}

def combine_risk(risky):
    """Merge the model's risky pair outputs into combined multi-drug combinations.
    Drugs linked through risky pairs form one combination (A+B and B+C become A+B+C).
    Side-effect probabilities are merged with noisy-OR: 1 - product(1 - p)."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for v in risky.values():
        a, b = find(v["drug1"]), find(v["drug2"])
        if a != b:
            parent[a] = b
    groups = {}
    for v in risky.values():
        groups.setdefault(find(v["drug1"]), []).append(v)

    combos = []
    for pairs in groups.values():
        pairs = sorted(pairs, key=lambda p: -p.get("risk_score", 0))
        drugs = sorted({d for p in pairs for d in (p["drug1"], p["drug2"])})
        sev = max((p.get("severity", "Moderate") for p in pairs), key=lambda s: RANK.get(s, 1))
        se = {}
        for p in pairs:
            for s in p.get("side_effects", []):
                e = se.setdefault(s["name"].lower(), {"name": s["name"].title(), "miss": 1.0, "drugs": set()})
                e["miss"] *= 1 - min(s["probability"], 0.99)
                e["drugs"].update([p["drug1"], p["drug2"]])
        effects = sorted(({"name": e["name"], "probability": 1 - e["miss"], "drugs": sorted(e["drugs"])}
                          for e in se.values()), key=lambda x: -x["probability"])
        contrib = {}
        for p in pairs:
            for me, other in ((p["drug1"], p["drug2"]), (p["drug2"], p["drug1"])):
                sc = p.get("risk_score", 0)
                if me not in contrib or sc > contrib[me]["score"]:
                    contrib[me] = {"drug": me, "score": sc, "partner": other}
        contrib = sorted(contrib.values(), key=lambda c: -c["score"])
        pathways = sorted({x for p in pairs for x in p.get("shared_enzymes", []) + p.get("shared_targets", [])})
        top3 = ", ".join(e["name"].lower() for e in effects[:3])
        advice = f"Monitor closely for {top3}." if top3 else "Monitor the patient closely."
        if sev == "Major":
            advice += " Consider dosage reduction or separating administration times, and review the combination with the prescriber."
        combos.append({"drugs": drugs, "severity": sev, "score": pairs[0].get("risk_score", 0), "effects": effects,
                       "contrib": contrib, "pathways": pathways, "pairs": pairs, "advice": advice})
    combos.sort(key=lambda c: (-RANK.get(c["severity"], 1), -c["score"]))
    return combos

def build_pdf(results, risky, combos, drugs, path, who="", profile=None):
    """Short report: medicines, overall risk, and for each risky combination the side effects + advice."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    except ImportError:
        # reportlab not installed: fall back to the project's own generator with only the risky pairs
        trimmed = dict(results)
        trimmed["pairwise"] = risky
        report_gen.generate_pdf(trimmed, results.get("drug_metadata", {}), path, patient_info=profile or {})
        return

    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=18, spaceAfter=4)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=12, spaceBefore=10, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    el = [Paragraph("Multi-Drug Interaction Prediction Report", title)]
    meta = f"Date: {datetime.now():%d %b %Y}"
    if who:
        meta = f"Patient: {escape(str(who))} &nbsp;|&nbsp; " + meta
    el += [Paragraph(meta, body)]
    if profile and profile.get("age"):
        el += [Paragraph(f"Age {profile['age']}, {escape(str(profile.get('sex','')))}", body)]
    el += [HRFlowable(width="100%", color=colors.lightgrey, spaceBefore=6, spaceAfter=6)]
    el += [Paragraph("Medicines checked", h), Paragraph(escape(", ".join(drugs)), body)]
    el += [Paragraph("Overall risk", h), Paragraph(escape(str(results.get("severity", "Minor"))), body)]
    el += [Paragraph("Risky combinations", h)]
    if not combos:
        el += [Paragraph("No risky interactions were found for these medicines.", body)]
    for c in combos:
        _, _, label, _ = severity_style({"severity": c["severity"], "risk_score": c["score"]})
        el += [Paragraph(f"<b>{escape(' + '.join(c['drugs']))}</b> - {label.title()}", body)]
        if c["effects"]:
            txt = ", ".join(f"{escape(e['name'])} ({e['probability']*100:.0f}%)" for e in c["effects"][:6])
            el += [Paragraph(f"Possible side effects: {txt}", body)]
        el += [Paragraph(f"Advice: {escape(c['advice'])}", body), Spacer(1, 5)]
    el += [Spacer(1, 8), Paragraph("Decision support only. Always confirm with a qualified clinician.", small)]
    doc.build(el)

def render_analysis(drugs, key, profile=None, title_hint="", show_alternatives=True):
    """Combined prediction + Explainable AI + Report. Used by both portals."""
    if len(drugs) < 2:
        st.info("Add at least 2 medications to run the multi-drug interaction prediction.")
        return
    stitch_ids = [d['stitch_id'] for d in drugs]
    names = {d['stitch_id']: d['generic'] for d in drugs}
    results = predictor.predict(stitch_ids, resolved_names=names)
    pairwise = results.get("pairwise", {})
    risky = risky_pairs(pairwise)
    combos = combine_risk(risky)
    all_names = [d['generic'] for d in drugs]
    involved = {x for c in combos for x in c["drugs"]}

    t_pred, t_xai, t_rep = st.tabs(["Predictions", "Explainable AI", "Report"])

    # ---------------- Predictions: combined risky combinations ----------------
    with t_pred:
        overall = results.get("severity", "Minor")
        st.markdown(f"<h3 style='font-weight:800;margin:.4rem 0'>Overall regimen risk: {overall}</h3>", unsafe_allow_html=True)
        if not combos:
            st.markdown('<div class="ddi ddi-green"><b>No risky combination or side effects detected for these medicines.</b></div>',
                        unsafe_allow_html=True)
        else:
            st.caption(f"{len(involved)} of {len(all_names)} medicines are involved in risky combinations.")
        for c in combos:
            cls, bcls, btxt, _ = severity_style({"severity": c["severity"], "risk_score": c["score"]})
            top = c["contrib"][0]
            chips = "".join(f'<span class="badge {bcls}" title="From: {", ".join(e["drugs"])}">{e["name"]} {e["probability"]*100:.0f}%</span>'
                            for e in c["effects"][:8])
            se_html = f'<div class="se-label">Combined possible side effects</div>{chips}' if chips else ""
            st.markdown(f"""<div class="ddi {cls}"><span class="badge {bcls}">{btxt}</span>
            <div class="ddi-title">{" + ".join(c["drugs"])}</div>
            <div>Strongest contributor: <b>{top["drug"]}</b> (with {top["partner"]})</div>
            {se_html}
            <div class="se-label">What to do</div><div>{c["advice"]}</div></div>""", unsafe_allow_html=True)
        safe = [n for n in all_names if n not in involved]
        if combos and safe:
            st.caption("No risky interaction found for: " + ", ".join(safe))
        if show_alternatives and results.get("alternatives"):
            st.markdown("#### Safer alternatives")
            for a in results["alternatives"]:
                st.markdown(f"""<div class="ddi ddi-green">Replace <b>{a['original_drug']}</b> with <b>{a['suggested_drug']}</b>
                ({a.get('class','Alternative')})<br><small>{a['rationale']}</small></div>""", unsafe_allow_html=True)

    # ---------------- Explainable AI: why each combination was flagged ----------------
    with t_xai:
        if not combos:
            st.info("Nothing to explain - no risky combination was predicted.")
        for idx, c in enumerate(combos):
            _, bcls, btxt, colr = severity_style({"severity": c["severity"], "risk_score": c["score"]})
            st.markdown(f"#### {' + '.join(c['drugs'])}  <span class='badge {bcls}'>{btxt}</span>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1.2], gap="medium")
            with c1:
                st.markdown("**Why it was flagged**")
                if c["pathways"]:
                    st.write("The medicines act on the same body pathway(s): " + ", ".join(c["pathways"]) + ".")
                st.markdown("**Which drugs drive the risk**")
                for k in c["contrib"]:
                    st.write(f"- {k['drug']}: strongest interaction {k['score']*100:.0f}% (with {k['partner']})")
                st.markdown("**Interacting pairs inside this combination**")
                for p in c["pairs"]:
                    via = ", ".join(p.get("shared_enzymes", []) + p.get("shared_targets", []))
                    st.write(f"- {p['drug1']} + {p['drug2']}: {p.get('severity','Moderate')}, {p.get('risk_score',0)*100:.0f}%"
                             + (f" (via {via})" if via else ""))
            with c2:
                se = c["effects"][:6]
                if se:
                    st.markdown("**Combined side-effect likelihood**")
                    se = list(reversed(se))
                    fig = go.Figure(go.Bar(x=[s['probability'] * 100 for s in se], y=[s['name'] for s in se],
                                           orientation='h', marker=dict(color=colr),
                                           text=[f"{s['probability']*100:.0f}%" for s in se], textposition='inside',
                                           customdata=[", ".join(s['drugs']) for s in se],
                                           hovertemplate="%{y}: %{x:.0f}%<br>From: %{customdata}<extra></extra>"))
                    fig.update_layout(height=60 + 38 * len(se), margin=dict(l=5, r=5, t=5, b=5),
                                      plot_bgcolor='#fff', paper_bgcolor='#fff', xaxis_title="Probability (%)")
                    st.plotly_chart(fig, use_container_width=True, key=f"xai_{key}_{idx}")
                    st.caption("Probabilities from each interacting pair are merged into one value per side effect.")
                else:
                    st.caption("No specific side effects were listed for this combination.")
            st.divider()

    # ---------------- Report ----------------
    with t_rep:
        pdf_path = os.path.join(tempfile.gettempdir(), f"MDIP_{key}_{len(stitch_ids)}.pdf")
        try:
            who = (profile or {}).get("patient", "")
            build_pdf(results, risky, combos, all_names, pdf_path, who=who, profile=profile)
            with open(pdf_path, "rb") as fp:
                st.download_button("Download Report (PDF)", fp.read(),
                                   file_name=f"Interaction_Report_{title_hint or 'Report'}_{datetime.now():%Y%m%d_%H%M}.pdf",
                                   mime="application/pdf", use_container_width=True, key=f"dl_{key}")
        except Exception as ex:
            st.warning(f"Could not generate PDF: {ex}")

# ==============================================================================
# AUTH SCREEN (black page, logo on top, white rounded card)
# ==============================================================================
if not S.user:
    st.markdown("""
    <style>
    .stApp {background:#000 !important;}
    .block-container {max-width:520px !important; padding-top:2.5rem !important;}
    .auth-logo {text-align:center; color:#00D639; font-weight:800; font-size:2rem; line-height:1.25; margin:6px 0 28px;}
    .auth-title {text-align:center; font-size:1.7rem; font-weight:700; color:#1a1a1a; margin:0 0 18px;}
    div[data-testid="stForm"] {background:#fff !important; border:none !important; border-radius:28px !important;
      padding:34px 40px 30px !important; box-shadow:none !important;}
    div[data-testid="stForm"] label p {font-weight:700 !important; color:#444 !important; font-size:1rem !important;}
    div[data-testid="stForm"] div[data-baseweb="input"], div[data-testid="stForm"] div[data-baseweb="select"] > div {
      background:#fff !important; border:1px solid #8AA0B8 !important; border-radius:6px !important;}
    div[data-testid="stForm"] input {background:#fff !important; color:#111 !important;}
    div[data-testid="stForm"] div[data-baseweb="select"] * {color:#111 !important;}
    div[data-testid="stForm"] button {background:#027B45 !important; color:#fff !important; border:none !important;
      border-radius:999px !important; height:3.1rem; font-weight:600; box-shadow:none !important; margin-top:14px;}
    div[data-testid="stForm"] button p {color:#fff !important;}
    .st-key-switch_mode {text-align:center;}
    .st-key-switch_mode button {background:transparent !important; border:none !important; box-shadow:none !important;
      color:#00D639 !important; text-decoration:underline; font-weight:600; margin:10px auto 0; display:block;}
    .st-key-switch_mode button p {color:#00D639 !important;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-logo">Multi-Drug Interaction<br>Prediction</div>', unsafe_allow_html=True)

    if S.flash:
        st.success(S.flash)
        S.flash = ""

    if S.auth_mode == "login":
        with st.form("login_form"):
            st.markdown('<div class="auth-title">Log in</div>', unsafe_allow_html=True)
            role_l = st.selectbox("I am a", ["Doctor", "Patient"], key="role_l")
            em = st.text_input("Email Address")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", use_container_width=True):
                u = verify_login(em, pw, role_l.lower())
                if u:
                    S.user = u; S.own_drugs = []
                    st.rerun()
                else:
                    st.error("Invalid email, password, or role selection.")
        if st.button("Create an account", key="switch_mode"):
            S.auth_mode = "signup"; st.rerun()
    else:
        with st.form("signup_form"):
            st.markdown('<div class="auth-title">Create account</div>', unsafe_allow_html=True)
            role_s = st.selectbox("I am a", ["Doctor", "Patient"], key="role_s")
            nm = st.text_input("Full Name")
            em2 = st.text_input("Email Address", key="em2")
            p1 = st.text_input("Password (min 6 characters)", type="password", key="p1")
            p2 = st.text_input("Confirm Password", type="password", key="p2")
            if st.form_submit_button("Create account", use_container_width=True):
                if not nm.strip() or "@" not in em2:
                    st.error("Enter a valid name and email.")
                elif len(p1) < 6:
                    st.error("Password must be at least 6 characters.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                elif create_user(nm, em2, p1, role_s.lower()):
                    S.flash = "Account created. Please log in."
                    S.auth_mode = "login"
                    st.rerun()
                else:
                    st.error("This email is already registered.")
        if st.button("Back to log in", key="switch_mode"):
            S.auth_mode = "login"; st.rerun()
    st.stop()

# ==============================================================================
# NAVBAR (after login)
# ==============================================================================
user = S.user
c1, c2 = st.columns([4, 1.2])
c1.markdown(f"""<div class="nav"><div class="nav-title">Multi-Drug Interaction Prediction</div>
    <span class="pill">{user["name"]} &middot; {user["role"].title()}</span></div>""", unsafe_allow_html=True)
with c2:
    st.write("")
    if st.button("Logout", use_container_width=True):
        logout()

available_drugs = resolver.get_available_drugs()

# ==============================================================================
# DOCTOR PORTAL
# ==============================================================================
if user["role"] == "doctor":
    patients = list_patients()
    with db() as con:
        n_rx = con.execute("SELECT COUNT(*) FROM prescriptions WHERE doctor_id=?", (user["id"],)).fetchone()[0]
    st.markdown(f"""<div class="banner banner-doc"><div><h2>Doctor Dashboard</h2>
    <p>Screen drug combinations for yourself and manage prescriptions for your patients.</p>
    <p style="margin-top:8px"><b>Dr. {user['name'].replace('Dr. ','')}</b> &middot; Doctor</p></div>
    <div class="avatar">Dr</div></div>""", unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    kpi(k1, "Registered Patients", len(patients), "Available to prescribe", "k-blue")
    kpi(k2, "My Prescriptions", n_rx, "Tablets prescribed by me", "k-amber")
    st.write("")

    part1, part2 = st.tabs(["Part 1 · My Own Use (Drug Screener)", "Part 2 · Patient Prescriptions"])

    # ---------------- PART 1: doctor's own use ----------------
    with part1:
        head("Prescription Multi-Drug Screener", "h-green", "Add / remove tablets and see predictions")
        c1, c2 = st.columns([4, 1])
        pick = c1.selectbox("Medication", [""] + available_drugs, key=f"own_pick_{S.add_ctr}",
                            label_visibility="collapsed", placeholder="Search generic or brand name...")
        if c2.button("Add tablet", use_container_width=True, type="primary", key="own_add"):
            if pick:
                r = resolver.resolve(pick)
                if r.get("resolved") and r["generic"] not in S.own_drugs:
                    S.own_drugs.append(r["generic"])
                S.add_ctr += 1
                st.rerun()
        if S.own_drugs:
            for i, g in enumerate(list(S.own_drugs)):
                a, b = st.columns([6, 1])
                a.markdown(f'<div class="rx-row"><span><b>{g}</b></span></div>', unsafe_allow_html=True)
                if b.button("Remove", key=f"own_rm_{i}", use_container_width=True):
                    S.own_drugs.pop(i); st.rerun()
            if st.button("Clear all", key="own_clear"):
                S.own_drugs = []; st.rerun()
        else:
            st.caption("No tablets added yet.")
        with st.expander("Patient clinical factors (for contraindication check & report)"):
            e1, e2, e3, e4 = st.columns(4)
            age = e1.number_input("Age", 1, 110, 65)
            sex = e2.selectbox("Sex", ["Male", "Female"])
            renal = e3.selectbox("Renal", ["Normal (eGFR >= 60 mL/min)", "Moderate Insufficiency (30-59)", "Severe Impairment / ESRD (< 30)"])
            liver = e4.selectbox("Hepatic", ["Normal Hepatic Function", "Mild Impairment (Child-Pugh A)", "Moderate to Severe (Child-Pugh B/C)"])
            f1, f2 = st.columns(2)
            pud = f1.checkbox("Peptic ulcer / GI bleed"); asthma = f2.checkbox("Asthma / COPD")
        end()

        own = resolve_list(S.own_drugs)
        for d in own:
            gl, dc = d['generic'].lower(), d.get('drug_class', '').lower()
            if pud and ("nsaid" in dc or gl in ("aspirin", "ibuprofen", "naproxen")):
                st.error(f"Warning - peptic ulcer contraindication: **{d['generic']}** (COX-1 inhibition, GI hemorrhage risk)")
            if asthma and ("nsaid" in dc or gl == "aspirin"):
                st.error(f"Warning - aspirin-exacerbated respiratory disease risk: **{d['generic']}**")
            if "severe" in renal.lower() and gl == "metformin":
                st.error("Warning - metformin with eGFR < 30: lactic acidosis risk")
        render_analysis(own, "own", {"age": age, "sex": sex, "renal": renal, "hepatic": liver}, "Doctor")

    # ---------------- PART 2: patient prescriptions ----------------
    with part2:
        if not patients:
            st.warning("No patients have registered yet. Patients must create an account first.")
        else:
            head("Select Registered Patient", "h-blue", f"{len(patients)} registered")
            labels = {f"{p['name']} ({p['email']})": p for p in patients}
            sel = labels[st.selectbox("Patient name", list(labels.keys()), label_visibility="collapsed")]
            end()

            head(f"Prescribe tablets to {sel['name']}", "h-purple", "Saved to the patient's portal")
            c1, c2 = st.columns([3, 3])
            drug_pick = c1.selectbox("Tablet", [""] + available_drugs, key=f"rx_pick_{sel['id']}_{S.add_ctr}",
                                     placeholder="Search tablet...")
            note = c2.text_input("Dosage / instructions (optional)", key=f"rx_note_{sel['id']}_{S.add_ctr}",
                                 placeholder="e.g. 75 mg once daily after food")
            if st.button("Add tablet to patient", type="primary", key=f"rx_add_{sel['id']}"):
                if drug_pick:
                    r = resolver.resolve(drug_pick)
                    existing = [x['drug'].lower() for x in get_rx(sel['id']) if x['doctor_id'] == user['id']]
                    if r.get("resolved") and r['generic'].lower() not in existing:
                        add_rx(sel['id'], user['id'], r['generic'], note)
                        S.add_ctr += 1
                        st.rerun()
                    else:
                        st.info("Already prescribed by you, or drug not resolved.")
            end()

            rxs = get_rx(sel['id'])
            head(f"Current prescriptions of {sel['name']}", "h-green", f"{len(rxs)} tablet(s) from all doctors")
            if not rxs:
                st.caption("No tablets prescribed yet.")
            for rx in rxs:
                a, b = st.columns([6, 1])
                mine = rx['doctor_id'] == user['id']
                tag = '<span class="badge b-blue">Prescribed by you</span>' if mine else f'<span class="badge b-amber">Dr. {rx["doctor_name"].replace("Dr. ","")}</span>'
                a.markdown(f'<div class="rx-row"><span><b>{rx["drug"]}</b> <small style="color:#64748B">{rx["note"] or ""} · {rx["created"]}</small></span>{tag}</div>',
                           unsafe_allow_html=True)
                if mine and b.button("Remove", key=f"rxrm_{rx['id']}", use_container_width=True):
                    remove_rx(rx['id'], user['id']); st.rerun()
            end()

            regimen = resolve_list(sorted({r['drug'] for r in rxs}))
            st.markdown("### Combined regimen analysis (all doctors)")
            render_analysis(regimen, f"pt{sel['id']}", {"patient": sel['name']}, sel['name'].replace(" ", "_"))

# ==============================================================================
# PATIENT PORTAL
# ==============================================================================
else:
    rxs = get_rx(user["id"])
    docs = sorted({r['doctor_name'] for r in rxs})
    st.markdown(f"""<div class="banner banner-pat"><div><h2>My Health Dashboard</h2>
    <p>Every tablet prescribed to you, checked for risky combinations.</p>
    <p style="margin-top:8px"><b>{user['name']}</b> &middot; Patient</p></div>
    <div class="avatar">{user['name'][:1].upper()}</div></div>""", unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    kpi(k1, "My Tablets", len(rxs), "Active prescriptions", "k-purple")
    kpi(k2, "Doctors", len(docs), "Have prescribed to me", "k-blue")
    st.write("")

    t1, t2 = st.tabs(["My Prescriptions", "Predictions · Explainable AI · Reports"])
    with t1:
        head("Tablets prescribed by my doctors", "h-purple", f"{len(rxs)} total")
        if not rxs:
            st.info("No tablets have been prescribed yet. Ask your doctor to add them using your registered email: " + user["email"])
        for rx in rxs:
            st.markdown(f"""<div class="rx-row"><span><b>{rx['drug']}</b>
            <small style="color:#64748B"> {rx['note'] or ''}</small></span>
            <span><span class="badge b-amber">Dr. {rx['doctor_name'].replace('Dr. ','')}</span>
            <small style="color:#64748B"> {rx['created']}</small></span></div>""", unsafe_allow_html=True)
        end()
    with t2:
        if len(rxs) < 2:
            st.info("You need at least 2 prescribed tablets for an interaction check.")
        else:
            doc_filter = st.selectbox("Check tablets from", ["All doctors combined"] + docs)
            sel_rx = rxs if doc_filter.startswith("All") else [r for r in rxs if r['doctor_name'] == doc_filter]
            regimen = resolve_list(sorted({r['drug'] for r in sel_rx}))
            render_analysis(regimen, "mine", {"patient": user['name']}, user['name'].replace(" ", "_"),
                            show_alternatives=False)

st.markdown("<div style='text-align:center;color:#64748B;font-size:.8rem;margin-top:2rem'>Multi-Drug Interaction Prediction System<br>Decision support only - always confirm with a qualified clinician.</div>", unsafe_allow_html=True)
