"""Visual system for the FlightRisk Streamlit product surface."""

CSS = r"""
<style>
:root {
  --fr-bg: #f3efe6;
  --fr-surface: #fbfaf7;
  --fr-surface-2: #f7f3eb;
  --fr-ink: #101b2d;
  --fr-muted: #5d6673;
  --fr-line: #d9d2c5;
  --fr-navy: #17304f;
  --fr-navy-2: #274a70;
  --fr-amber: #bb7a24;
  --fr-amber-soft: #f1e2c6;
  --fr-teal: #3f716d;
  --fr-red: #934a4d;
  --fr-shadow: 0 14px 36px rgba(16, 27, 45, .08);
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  color: var(--fr-ink);
  background:
    linear-gradient(rgba(23, 48, 79, .025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(23, 48, 79, .025) 1px, transparent 1px),
    var(--fr-bg);
  background-size: 42px 42px;
}

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { max-width: 1220px; padding-top: .45rem; padding-bottom: 3rem; }
h1, h2, h3, h4, p, div, span, label { color: var(--fr-ink); }

[data-testid="stForm"] {
  background: var(--fr-surface);
  border: 1px solid var(--fr-line);
  border-radius: 12px;
  padding: 1rem;
  box-shadow: var(--fr-shadow);
}

[data-testid="stMetric"] {
  background: var(--fr-surface);
  border: 1px solid var(--fr-line);
  border-radius: 10px;
  padding: .75rem .82rem;
  box-shadow: 0 8px 22px rgba(16, 27, 45, .055);
}
[data-testid="stMetric"] label { color: var(--fr-muted) !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { color: var(--fr-ink) !important; font-variant-numeric: tabular-nums; }

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  background: var(--fr-navy) !important;
  color: #fff !important;
  border: 1px solid var(--fr-navy) !important;
  border-radius: 8px !important;
  min-height: 2.45rem;
  font-weight: 800 !important;
  box-shadow: none !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
  background: var(--fr-navy-2) !important;
  border-color: var(--fr-navy-2) !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
  background: #fff;
  border: 1px solid var(--fr-line);
  border-radius: 8px;
  color: var(--fr-ink);
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--fr-surface) !important;
  border: 1px dashed #aaa294 !important;
  border-radius: 10px !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--fr-line);
  border-radius: 10px;
  overflow: hidden;
}

button[data-baseweb="tab"] {
  border-radius: 8px 8px 0 0;
  font-weight: 780;
}

.fr-topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: .72rem .9rem;
  background: rgba(251, 250, 247, .94);
  border: 1px solid var(--fr-line);
  border-radius: 10px;
  box-shadow: 0 7px 20px rgba(16, 27, 45, .05);
}
.fr-brand { display: flex; align-items: center; gap: .7rem; }
.fr-mark {
  width: 31px; height: 31px; border-radius: 8px;
  display: grid; place-items: center;
  color: #fff; background: var(--fr-navy);
  font-weight: 950; letter-spacing: -.05em;
}
.fr-brand-name { font-weight: 930; letter-spacing: .105em; font-size: .96rem; }
.fr-byline { color: var(--fr-muted); font-size: .77rem; margin-top: .05rem; }
.fr-status { display: flex; gap: .42rem; flex-wrap: wrap; justify-content: flex-end; }
.fr-chip {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .28rem .5rem; border-radius: 999px;
  background: var(--fr-surface-2); border: 1px solid var(--fr-line);
  color: var(--fr-muted); font-size: .72rem; font-weight: 800;
}
.fr-chip.ok { color: var(--fr-teal); }

.fr-hero {
  margin: .72rem 0 .85rem;
  padding: 1.35rem;
  border: 1px solid var(--fr-line);
  border-radius: 14px;
  background:
    radial-gradient(circle at 92% 10%, rgba(187, 122, 36, .12), transparent 32%),
    linear-gradient(145deg, #fbfaf7 0%, #f6f1e7 100%);
  box-shadow: var(--fr-shadow);
}
.fr-kicker {
  color: var(--fr-amber); font-size: .73rem; font-weight: 920;
  letter-spacing: .115em; text-transform: uppercase;
}
.fr-title {
  margin: .55rem 0 .52rem;
  max-width: 740px;
  font-size: clamp(2rem, 4vw, 4.15rem);
  line-height: .98;
  letter-spacing: -.058em;
  font-weight: 950;
}
.fr-title em { color: var(--fr-navy-2); font-style: normal; }
.fr-subtitle { max-width: 700px; color: var(--fr-muted); font-size: 1rem; line-height: 1.62; }
.fr-constraint {
  margin-top: .85rem; padding: .67rem .78rem;
  border-left: 3px solid var(--fr-amber);
  background: rgba(241, 226, 198, .46);
  color: #62533b; font-size: .83rem; line-height: 1.45;
}

.fr-flight-card {
  background: var(--fr-navy);
  border-radius: 12px;
  padding: 1rem;
  min-height: 245px;
  box-shadow: 0 18px 42px rgba(16, 27, 45, .19);
}
.fr-flight-card * { color: #f8f5ed; }
.fr-flight-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.fr-flight-id { font-size: .75rem; letter-spacing: .13em; text-transform: uppercase; opacity: .72; font-weight: 850; }
.fr-route { font-size: 1.75rem; letter-spacing: -.04em; font-weight: 930; margin-top: .18rem; }
.fr-priority {
  padding: .3rem .48rem; border-radius: 6px;
  background: var(--fr-amber); color: #fff !important;
  font-size: .69rem; letter-spacing: .08em; font-weight: 900;
}
.fr-risk-number { margin-top: 1.15rem; font-size: 3.15rem; line-height: 1; font-weight: 950; font-variant-numeric: tabular-nums; }
.fr-risk-label { font-size: .75rem; opacity: .68; margin-top: .24rem; }
.fr-flight-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .55rem; margin-top: 1rem; }
.fr-flight-stat { border-top: 1px solid rgba(255,255,255,.17); padding-top: .48rem; }
.fr-flight-stat b { display: block; font-size: .96rem; }
.fr-flight-stat span { display: block; opacity: .62; font-size: .69rem; margin-top: .08rem; }

.fr-section-head { margin: .3rem 0 .72rem; }
.fr-section-head h2 { margin: 0; font-size: 1.32rem; letter-spacing: -.026em; }
.fr-section-head p { margin: .25rem 0 0; color: var(--fr-muted); line-height: 1.5; font-size: .9rem; }
.fr-panel {
  padding: .88rem .95rem; background: var(--fr-surface);
  border: 1px solid var(--fr-line); border-radius: 10px;
  box-shadow: 0 8px 22px rgba(16, 27, 45, .05);
}
.fr-decision {
  padding: .92rem 1rem; background: var(--fr-navy);
  border-radius: 10px; margin-bottom: .72rem;
}
.fr-decision * { color: #fff; }
.fr-decision-kicker { color: #e7c58c !important; font-size: .7rem; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
.fr-decision h3 { margin: .25rem 0 .15rem; font-size: 1.32rem; }
.fr-decision p { margin: 0; opacity: .72; font-size: .84rem; }
.fr-context-row {
  display: flex; justify-content: space-between; gap: 1rem;
  padding: .62rem 0; border-bottom: 1px solid var(--fr-line);
}
.fr-context-row:last-child { border-bottom: 0; }
.fr-context-label { font-weight: 760; }
.fr-context-meta { color: var(--fr-muted); font-size: .76rem; margin-top: .1rem; }
.fr-context-value { font-weight: 900; font-variant-numeric: tabular-nums; white-space: nowrap; }
.fr-note {
  padding: .7rem .8rem; border: 1px solid var(--fr-line); border-radius: 9px;
  background: var(--fr-surface-2); color: var(--fr-muted); font-size: .81rem; line-height: 1.48;
}
.fr-validation-card {
  background: var(--fr-surface); border: 1px solid var(--fr-line);
  border-radius: 10px; padding: .88rem; min-height: 108px;
}
.fr-validation-card b { display: block; font-size: 1.05rem; }
.fr-validation-card span { display: block; color: var(--fr-muted); font-size: .79rem; margin-top: .25rem; line-height: 1.42; }
.fr-footer { text-align: center; color: var(--fr-muted); font-size: .76rem; margin-top: 1.4rem; }

[data-testid="stExpander"] {
  background: var(--fr-surface) !important;
  border: 1px solid var(--fr-line) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}

@media (max-width: 900px) {
  .fr-title { font-size: 2.45rem; }
  .fr-status { display: none; }
  .fr-flight-grid { grid-template-columns: 1fr; }
}
</style>
"""
