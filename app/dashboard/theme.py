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
[data-testid="stMainBlockContainer"],
.main .block-container {
  width: min(1500px, calc(100% - 2rem));
  max-width: 1500px;
  padding-top: .85rem !important;
  padding-bottom: 3rem;
}
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
.stButton > button *,
.stFormSubmitButton > button *,
.stDownloadButton > button * {
  color: #fff !important;
  fill: #fff !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
  background: var(--fr-navy-2) !important;
  border-color: var(--fr-navy-2) !important;
}
.stButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
  outline: 3px solid rgba(35, 101, 143, .28) !important;
  outline-offset: 2px;
}
.stButton > button:disabled,
.stFormSubmitButton > button:disabled,
.stDownloadButton > button:disabled {
  background: #d7e4ef !important;
  border-color: #c3d5e4 !important;
  color: #52677b !important;
  opacity: 1 !important;
}
.stButton > button:disabled *,
.stFormSubmitButton > button:disabled *,
.stDownloadButton > button:disabled * {
  color: #52677b !important;
  fill: #52677b !important;
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
.fr-source-link,
.fr-source-link * {
  color: var(--fr-navy-2) !important;
}
.fr-source-link {
  display: inline-flex;
  align-items: center;
  gap: .28rem;
  margin-top: .13rem;
  font-size: .68rem;
  line-height: 1.2;
  font-weight: 820;
  text-decoration: none;
}
.fr-source-link:hover { color: var(--fr-navy) !important; text-decoration: underline; }
.fr-source-link span { font-size: .72rem; transform: translateY(-.02rem); }
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
  padding: 1.18rem;
  border: 1px solid var(--fr-line);
  border-radius: 10px;
  background: linear-gradient(135deg, #ffffff 0%, #f2f9ff 58%, #eaf5fe 100%);
  box-shadow: var(--fr-shadow);
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(430px, .92fr);
  gap: 1.25rem;
  align-items: stretch;
}
.fr-hero-copy {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: .5rem .35rem .55rem .2rem;
}
.fr-kicker {
  color: var(--fr-amber); font-size: .72rem; font-weight: 920;
  letter-spacing: .12em; text-transform: uppercase;
}
.fr-title {
  margin: .5rem 0 .48rem;
  max-width: 760px;
  font-size: clamp(2.35rem, 3.65vw, 4rem);
  line-height: .99;
  letter-spacing: -.056em;
  font-weight: 950;
}
.fr-title em { color: var(--fr-navy-2); font-style: normal; }
.fr-subtitle { max-width: 720px; color: var(--fr-muted); font-size: .98rem; line-height: 1.58; }
.fr-constraint {
  margin-top: .78rem; padding: .62rem .75rem;
  border-left: 3px solid var(--fr-amber);
  background: rgba(237, 246, 255, .88);
  color: #425a70; font-size: .82rem; line-height: 1.45;
}

.fr-coverage-visual {
  position: relative;
  min-height: 430px;
  border: 1px solid rgba(191, 216, 236, .88);
  border-radius: 10px;
  overflow: hidden;
  background:
    radial-gradient(circle at 76% 8%, rgba(255,255,255,.92), transparent 34%),
    linear-gradient(155deg, rgba(228, 242, 253, .78), rgba(247, 251, 255, .94));
}
.fr-coverage-visual::after {
  content: "";
  position: absolute;
  inset: 45% 0 0;
  background: linear-gradient(to bottom, transparent, rgba(238,247,255,.92) 38%, #edf7ff 100%);
  pointer-events: none;
}
.fr-coverage-head {
  position: relative;
  z-index: 3;
  display: flex;
  justify-content: space-between;
  gap: .9rem;
  align-items: flex-start;
  padding: .82rem .9rem 0;
}
.fr-coverage-head span {
  display: block;
  color: var(--fr-navy-2);
  font-size: .62rem;
  font-weight: 920;
  letter-spacing: .12em;
}
.fr-coverage-head strong {
  display: block;
  margin-top: .14rem;
  color: var(--fr-ink);
  font-size: .98rem;
  font-weight: 920;
}
.fr-coverage-head p {
  max-width: 185px;
  margin: 0;
  color: var(--fr-muted);
  font-size: .65rem;
  line-height: 1.35;
  text-align: right;
}
.fr-map-frame {
  position: absolute;
  z-index: 1;
  inset: 2.65rem -.3rem 5.6rem .2rem;
}
.fr-coverage-svg { width: 100%; height: 100%; overflow: visible; }
.fr-map-land { fill: rgba(194, 220, 239, .68); }
.fr-map-states { fill: none; stroke: rgba(255,255,255,.88); stroke-width: 1.2; vector-effect: non-scaling-stroke; }
.airport-dot { fill: #2e78a5; opacity: .62; stroke: rgba(255,255,255,.9); stroke-width: .45; vector-effect: non-scaling-stroke; }
.airport-dot-major { fill: #b7791f; opacity: .95; stroke-width: .75; }
.fr-territory-key rect { fill: rgba(245,250,255,.82); stroke: rgba(146,187,217,.58); stroke-width: 1; }
.fr-territory-key text { fill: #5d7890; font-size: 12px; font-weight: 850; letter-spacing: 1.25px; }
.fr-airport-labels text {
  fill: #164a73;
  font-size: 13px;
  font-weight: 900;
  letter-spacing: .72px;
  paint-order: stroke;
  stroke: rgba(245, 250, 255, .96);
  stroke-width: 3.4px;
  stroke-linejoin: round;
  pointer-events: none;
}
.fr-coverage-visual .fr-airport-labels text {
  font-size: 17px;
  stroke-width: 4px;
}
.fr-coverage-visual .airport-dot-major { r: 4.4px; }

.fr-heatmap-shell {
  margin: .48rem 0 1.15rem;
  padding: 1.08rem 1.15rem .72rem;
  border: 1px solid rgba(82, 160, 209, .4);
  border-radius: 12px;
  background:
    radial-gradient(circle at 78% 5%, rgba(52, 132, 181, .34), transparent 34%),
    linear-gradient(148deg, #0d304a 0%, #123f5f 54%, #0b2a42 100%);
  box-shadow: 0 20px 46px rgba(11, 47, 73, .19);
  overflow: hidden;
}
.fr-heatmap-intro {
  position: relative;
  z-index: 2;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1.25rem;
  align-items: start;
}
.fr-heatmap-eyebrow {
  display: block;
  color: #76cef1 !important;
  font-size: .66rem;
  font-weight: 920;
  letter-spacing: .14em;
}
.fr-heatmap-intro strong {
  display: block;
  margin-top: .22rem;
  color: #fffaf0;
  font-size: clamp(1.45rem, 2.3vw, 2rem);
  line-height: 1.05;
  letter-spacing: -.035em;
}
.fr-heatmap-intro p {
  max-width: 720px;
  margin: .45rem 0 0;
  color: rgba(226, 240, 249, .76);
  font-size: .82rem;
  line-height: 1.48;
}
.fr-heatmap-meta {
  display: flex;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: .38rem;
  max-width: 410px;
}
.fr-heatmap-meta span,
.fr-heatmap-meta em {
  padding: .3rem .52rem;
  border: 1px solid rgba(145, 205, 238, .25);
  border-radius: 999px;
  background: rgba(225, 244, 255, .08);
  color: rgba(241, 248, 252, .88);
  font-size: .68rem;
  font-style: normal;
  font-weight: 800;
}
.fr-heatmap-meta em {
  width: 100%;
  border-color: rgba(242, 184, 75, .35);
  color: #f5cc7d;
  text-align: center;
}
.fr-heatmap-map {
  width: 100%;
  height: 520px;
  display: flex;
  justify-content: center;
}
.fr-heatmap-svg { width: 100%; height: 100%; overflow: visible; }
.fr-heatmap-shell .fr-map-land { fill: #235a7b; fill-opacity: .82; }
.fr-heatmap-shell .fr-map-states { stroke: rgba(233, 245, 252, .34); stroke-width: 1; }
.fr-heatmap-shell .fr-territory-key rect {
  fill: rgba(8, 37, 58, .68);
  stroke: rgba(139, 199, 232, .35);
}
.fr-heatmap-shell .fr-territory-key text { fill: rgba(221, 238, 248, .62); }
.fr-heatmap-shell .fr-airport-labels text {
  fill: #fff8e9;
  font-size: 15px;
  stroke: rgba(10, 43, 67, .96);
  stroke-width: 4px;
}
.fr-heat-dot {
  pointer-events: none;
  vector-effect: non-scaling-stroke;
}
.fr-heat-hit {
  fill: none;
  stroke: none;
  pointer-events: all;
  cursor: default;
  transition: fill .12s ease, stroke .12s ease;
  vector-effect: non-scaling-stroke;
}
.fr-heat-hover:hover .fr-heat-hit {
  fill: rgba(255, 255, 255, .08);
  stroke: rgba(255, 255, 255, .78);
  stroke-width: 1.5;
}
.fr-heat-tooltip {
  opacity: 0;
  pointer-events: none;
  transition: opacity .12s ease;
}
.fr-heat-hover:hover .fr-heat-tooltip { opacity: 1; }
.fr-heat-tooltip rect {
  fill: rgba(6, 29, 46, .97);
  stroke: rgba(145, 216, 246, .65);
  stroke-width: 1;
  filter: drop-shadow(0 5px 9px rgba(0, 0, 0, .32));
}
.fr-heat-tooltip text {
  fill: rgba(232, 244, 251, .82);
  stroke: none;
  paint-order: normal;
  font-size: 11px;
  letter-spacing: 0;
}
.fr-heat-tooltip .fr-heat-tooltip-code {
  fill: #fffaf0;
  font-size: 13px;
  font-weight: 920;
  letter-spacing: .8px;
}
.fr-heat-tooltip .fr-heat-tooltip-rate { fill: #f5cc7d; font-weight: 820; }
.fr-heat-tooltip .fr-heat-tooltip-support { font-size: 10px; }
.fr-heatmap-legend {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: .42rem .72rem;
  padding-top: .5rem;
  border-top: 1px solid rgba(139, 199, 232, .22);
  color: rgba(222, 238, 247, .72);
  font-size: .7rem;
}
.fr-heatmap-legend b { color: #fffaf0; font-size: .72rem; }
.fr-heatmap-legend span { display: inline-flex; align-items: center; gap: .28rem; }
.fr-heatmap-legend i {
  display: inline-block;
  width: .58rem;
  height: .58rem;
  border-radius: 50%;
  box-shadow: 0 0 0 1px rgba(255,255,255,.7), 0 0 8px rgba(255,255,255,.12);
}
.fr-heatmap-legend em { color: rgba(222, 238, 247, .68); font-style: normal; }

.fr-flight-card {
  position: absolute;
  z-index: 4;
  left: .82rem;
  right: .82rem;
  bottom: .82rem;
  background: rgba(18, 72, 111, .965);
  border-radius: 9px;
  padding: .82rem .88rem;
  min-height: 0;
  box-shadow: 0 18px 38px rgba(20, 74, 115, .18);
  backdrop-filter: blur(8px);
}
.fr-flight-card * { color: #f8f5ed; }
.fr-flight-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.fr-flight-id { font-size: .7rem; letter-spacing: .13em; text-transform: uppercase; opacity: .72; font-weight: 850; }
.fr-route { font-size: 1.42rem; letter-spacing: -.04em; font-weight: 930; margin-top: .12rem; }
.fr-priority {
  padding: .3rem .48rem; border-radius: 4px;
  background: var(--fr-amber); color: #fff !important;
  font-size: .68rem; letter-spacing: .08em; font-weight: 900;
}
.fr-risk-row { display: grid; grid-template-columns: .62fr 1.38fr; gap: .9rem; align-items: end; margin-top: .65rem; }
.fr-risk-number { margin-top: 0; font-size: 2.45rem; line-height: 1; font-weight: 950; font-variant-numeric: tabular-nums; }
.fr-risk-label { font-size: .73rem; opacity: .68; margin-top: .24rem; }
.fr-flight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .48rem; margin-top: 0; }
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
  .fr-hero { grid-template-columns: 1fr; }
  .fr-coverage-visual { min-height: 420px; }
  .fr-risk-row { grid-template-columns: 1fr; }
  .fr-flight-grid { grid-template-columns: repeat(3, 1fr); }
  .fr-workflow { grid-template-columns: repeat(2, 1fr); }
  .fr-metric-grid.cols-2, .fr-metric-grid.cols-3, .fr-metric-grid.cols-4 { grid-template-columns: 1fr; }
  .fr-step:nth-child(2) { border-right: 0; }
  .fr-step:nth-child(-n+2) { border-bottom: 1px solid var(--fr-line); }
}

@media (max-width: 600px) {
  [data-testid="stMainBlockContainer"], .main .block-container { width: calc(100% - 1rem); }
  .fr-hero { padding: .82rem; }
  .fr-coverage-head { display: block; }
  .fr-coverage-head p { margin-top: .28rem; max-width: none; text-align: left; }
  .fr-flight-grid { grid-template-columns: 1fr; }
  .fr-coverage-visual { min-height: 510px; }
  .fr-heatmap-shell { padding: .82rem .72rem .62rem; }
  .fr-heatmap-intro { grid-template-columns: 1fr; gap: .7rem; }
  .fr-heatmap-meta { justify-content: flex-start; max-width: none; }
  .fr-heatmap-meta em { width: auto; }
  .fr-heatmap-map { height: 360px; }
  .fr-heatmap-legend { justify-content: flex-start; }
}
</style>
"""
