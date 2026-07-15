"""Flight Delay Risk visual system: calm aviation blue and operational clarity."""

CSS = r"""
<style>
:root {
  --fr-bg: #f4f9ff;
  --fr-surface: #ffffff;
  --fr-surface-2: #edf6ff;
  --fr-ink: #10243e;
  --fr-muted: #5d6f83;
  --fr-line: #d8e7f5;
  --fr-navy: #164a73;
  --fr-navy-2: #23658f;
  --fr-amber: #b7791f;
  --fr-amber-soft: #fff3d8;
  --fr-teal: #23756f;
  --fr-red: #9a4e57;
  --fr-shadow: 0 14px 34px rgba(28, 82, 124, .075);
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  color: var(--fr-ink);
  background: var(--fr-bg);
}

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { max-width: 1240px; padding-top: .5rem; padding-bottom: 3rem; }
h1, h2, h3, h4, p, div, span, label { color: var(--fr-ink); }

[data-testid="stForm"] {
  background: var(--fr-surface);
  border: 1px solid var(--fr-line);
  border-radius: 10px;
  padding: 1rem;
  box-shadow: var(--fr-shadow);
}

[data-testid="stMetric"] {
  background: var(--fr-surface);
  border: 1px solid var(--fr-line);
  border-radius: 8px;
  padding: .75rem .82rem;
  box-shadow: 0 7px 18px rgba(28, 82, 124, .045);
}
[data-testid="stMetric"] label { color: var(--fr-muted) !important; font-weight: 750 !important; }
[data-testid="stMetricValue"] { color: var(--fr-ink) !important; font-variant-numeric: tabular-nums; }

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  background: var(--fr-navy) !important;
  color: #fff !important;
  border: 1px solid var(--fr-navy) !important;
  border-radius: 6px !important;
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
  border-radius: 6px;
  color: var(--fr-ink);
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--fr-surface) !important;
  border: 1px dashed #9f9789 !important;
  border-radius: 8px !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--fr-line);
  border-radius: 8px;
  overflow: hidden;
}

button[data-baseweb="tab"] {
  border-radius: 0 !important;
  font-weight: 780;
  padding-left: 1.05rem !important;
  padding-right: 1.05rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--fr-navy) !important; }

.fr-topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: .7rem .85rem;
  background: var(--fr-surface);
  border: 1px solid var(--fr-line);
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(28, 82, 124, .045);
}
.fr-brand { display: flex; align-items: center; gap: .7rem; }
.fr-mark {
  width: 31px; height: 31px; border-radius: 6px;
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
.fr-chip.warn { color: var(--fr-red); }
.fr-chip.release { color: var(--fr-navy); background: var(--fr-amber-soft); border-color: #dec59c; }

.fr-hero {
  margin: .72rem 0 .65rem;
  padding: 1.4rem;
  border: 1px solid var(--fr-line);
  border-radius: 10px;
  background: linear-gradient(142deg, #ffffff 0%, #edf7ff 100%);
  box-shadow: var(--fr-shadow);
  position: relative;
  overflow: hidden;
}
.fr-hero::after {
  content: "";
  position: absolute;
  width: 260px; height: 260px;
  right: -120px; top: -150px;
  border: 46px solid rgba(35, 101, 143, .065);
  border-radius: 50%;
  pointer-events: none;
}
.fr-kicker {
  color: var(--fr-amber); font-size: .72rem; font-weight: 920;
  letter-spacing: .12em; text-transform: uppercase;
}
.fr-title {
  margin: .55rem 0 .52rem;
  max-width: 740px;
  font-size: clamp(2rem, 4vw, 4rem);
  line-height: .99;
  letter-spacing: -.056em;
  font-weight: 950;
}
.fr-title em { color: var(--fr-navy-2); font-style: normal; }
.fr-subtitle { max-width: 710px; color: var(--fr-muted); font-size: 1rem; line-height: 1.62; }
.fr-constraint {
  margin-top: .85rem; padding: .65rem .78rem;
  border-left: 3px solid var(--fr-amber);
  background: rgba(237, 246, 255, .88);
  color: #425a70; font-size: .82rem; line-height: 1.45;
}

.fr-flight-card {
  background: var(--fr-navy);
  border-radius: 9px;
  padding: 1rem;
  min-height: 245px;
  box-shadow: 0 18px 38px rgba(20, 74, 115, .18);
}
.fr-flight-card * { color: #f8f5ed; }
.fr-flight-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.fr-flight-id { font-size: .7rem; letter-spacing: .13em; text-transform: uppercase; opacity: .72; font-weight: 850; }
.fr-route { font-size: 1.75rem; letter-spacing: -.04em; font-weight: 930; margin-top: .18rem; }
.fr-priority {
  padding: .3rem .48rem; border-radius: 4px;
  background: var(--fr-amber); color: #fff !important;
  font-size: .68rem; letter-spacing: .08em; font-weight: 900;
}
.fr-risk-number { margin-top: 1.12rem; font-size: 3.15rem; line-height: 1; font-weight: 950; font-variant-numeric: tabular-nums; }
.fr-risk-label { font-size: .73rem; opacity: .68; margin-top: .24rem; }
.fr-flight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .55rem; margin-top: 1rem; }
.fr-flight-stat { border-top: 1px solid rgba(255,255,255,.17); padding-top: .48rem; }
.fr-flight-stat b { display: block; font-size: .96rem; }
.fr-flight-stat span { display: block; opacity: .62; font-size: .68rem; margin-top: .08rem; }

.fr-workflow {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid var(--fr-line);
  border-radius: 8px;
  overflow: hidden;
  margin: 0 0 .9rem;
  background: var(--fr-surface);
}
.fr-step { display: flex; align-items: center; gap: .55rem; padding: .62rem .72rem; border-right: 1px solid var(--fr-line); }
.fr-step:last-child { border-right: 0; }
.fr-step span { width: 22px; height: 22px; display: grid; place-items: center; border-radius: 50%; background: var(--fr-navy); color: #fff; font-size: .7rem; font-weight: 900; }
.fr-step b { font-size: .78rem; }

.fr-section-head { margin: .35rem 0 .72rem; }
.fr-section-head h2 { margin: 0; font-size: 1.34rem; letter-spacing: -.026em; }
.fr-section-head p { margin: .25rem 0 0; color: var(--fr-muted); line-height: 1.5; font-size: .9rem; }
.fr-panel {
  padding: .72rem .82rem; background: var(--fr-surface);
  border: 1px solid var(--fr-line); border-radius: 8px;
  box-shadow: 0 7px 18px rgba(28, 82, 124, .04);
}
.fr-decision {
  padding: .64rem .82rem; background: var(--fr-navy);
  border-radius: 8px; margin: .65rem 0 .62rem;
}
.fr-decision * { color: #fff; }
.fr-decision-kicker { color: #e7c58c !important; font-size: .69rem; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
.fr-decision h3 { margin: .15rem 0 .08rem; font-size: 1.08rem; }
.fr-decision p { margin: 0; opacity: .76; font-size: .79rem; line-height: 1.35; }
.fr-context-row {
  display: flex; justify-content: space-between; gap: 1rem;
  padding: .43rem 0; border-bottom: 1px solid var(--fr-line);
}
.fr-context-row:last-child { border-bottom: 0; }
.fr-context-label { font-weight: 760; }
.fr-context-meta { color: var(--fr-muted); font-size: .75rem; margin-top: .1rem; }
.fr-context-value { font-weight: 900; font-variant-numeric: tabular-nums; white-space: nowrap; }
.fr-note {
  padding: .7rem .8rem; border: 1px solid var(--fr-line); border-radius: 7px;
  background: var(--fr-surface-2); color: var(--fr-muted); font-size: .81rem; line-height: 1.48;
}
.fr-note.emphasis { border-left: 3px solid var(--fr-amber); color: var(--fr-ink); }
.fr-validation-card {
  background: var(--fr-surface); border: 1px solid var(--fr-line);
  border-radius: 8px; padding: .88rem; min-height: 88px;
}
.fr-validation-card b { display: block; font-size: 1.04rem; margin-top: .2rem; }
.fr-validation-card span { display: block; color: var(--fr-muted); font-size: .78rem; line-height: 1.42; }

.fr-contribution {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: .72rem .78rem; border-bottom: 1px solid var(--fr-line); background: var(--fr-surface);
}
.fr-contribution:first-of-type { margin-top: .65rem; border-radius: 8px 8px 0 0; border-top: 1px solid var(--fr-line); }
.fr-contribution:last-of-type { border-radius: 0 0 8px 8px; }
.fr-contribution > div:first-child b { display: block; font-size: .86rem; }
.fr-contribution > div:first-child span { display: block; color: var(--fr-muted); font-size: .73rem; margin-top: .08rem; }
.fr-contribution-value { font-size: .77rem; font-weight: 850; white-space: nowrap; }
.fr-contribution-value.up { color: var(--fr-amber); }
.fr-contribution-value.down { color: var(--fr-teal); }

.fr-metric-grid {
  display: grid;
  gap: .72rem;
  margin: .65rem 0 .8rem;
}
.fr-metric-grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.fr-metric-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.fr-metric-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.fr-metric-card {
  min-height: 142px;
  padding: .82rem .9rem;
  background: var(--fr-surface);
  border: 1px solid var(--fr-line);
  border-radius: 9px;
  box-shadow: 0 7px 18px rgba(28, 82, 124, .04);
}
.fr-metric-label {
  color: var(--fr-muted);
  font-size: .76rem;
  font-weight: 850;
  letter-spacing: .01em;
}
.fr-metric-value {
  margin-top: .3rem;
  color: var(--fr-ink);
  font-size: 1.75rem;
  line-height: 1.05;
  font-weight: 940;
  letter-spacing: -.035em;
  font-variant-numeric: tabular-nums;
}
.fr-metric-help {
  margin-top: .5rem;
  color: var(--fr-muted);
  font-size: .76rem;
  line-height: 1.42;
}
.fr-metric-direction {
  margin-top: .42rem;
  color: var(--fr-navy-2);
  font-size: .69rem;
  font-weight: 850;
  text-transform: uppercase;
  letter-spacing: .055em;
}
.fr-context-summary {
  margin: -.18rem 0 .72rem;
  padding: .48rem .65rem;
  color: var(--fr-muted);
  background: rgba(237, 246, 255, .72);
  border: 1px solid var(--fr-line);
  border-radius: 7px;
  font-size: .77rem;
  line-height: 1.4;
}
.fr-reliability-callout {
  margin: .58rem 0 .35rem;
  padding: .55rem .62rem;
  color: var(--fr-navy);
  background: var(--fr-surface-2);
  border-radius: 6px;
  font-size: .78rem;
  font-weight: 780;
}

.fr-model-badges { display: flex; flex-wrap: wrap; gap: .42rem; margin: .55rem 0 .72rem; }
.fr-model-badge {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .28rem .52rem; border-radius: 999px;
  background: #e7f3ff; border: 1px solid #c9e1f4;
  color: var(--fr-navy); font-size: .73rem; font-weight: 850;
}
.fr-model-badge b { color: var(--fr-navy); }
.fr-support-quality { color: var(--fr-muted); font-size: .72rem; font-weight: 800; }
.fr-footer { text-align: center; color: var(--fr-muted); font-size: .75rem; margin-top: 1.4rem; }

[data-testid="stExpander"] {
  background: var(--fr-surface) !important;
  border: 1px solid var(--fr-line) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}

@media (max-width: 900px) {
  .fr-title { font-size: 2.45rem; }
  .fr-status { display: none; }
  .fr-flight-grid { grid-template-columns: 1fr; }
  .fr-workflow { grid-template-columns: repeat(2, 1fr); }
  .fr-metric-grid.cols-2, .fr-metric-grid.cols-3, .fr-metric-grid.cols-4 { grid-template-columns: 1fr; }
  .fr-step:nth-child(2) { border-right: 0; }
  .fr-step:nth-child(-n+2) { border-bottom: 1px solid var(--fr-line); }
}
</style>
"""
