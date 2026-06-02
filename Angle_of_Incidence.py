import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import pvlib
import requests
import numpy as np
from datetime import datetime
import math
from streamlit_js_eval import get_geolocation
import xgboost as xgb
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Solar Forecast", layout="wide", page_icon="☀️")

# ─────────────────────────────────────────────
# 2. AUTO-REFRESH
# ─────────────────────────────────────────────
refresh_count = st_autorefresh(interval=300000, limit=1000, key="live_dashboard")

# ─────────────────────────────────────────────
# 3. MODEL LOAD (Updated for GitHub/Streamlit Cloud)
# ─────────────────────────────────────────────
@st.cache_resource
def load_master_brain():
    # Looks for the compressed model in the same folder on the cloud server
    model_path = "GLOBAL_MASTER_BRAIN_COMPRESSED.pkl"
    return joblib.load(model_path)

try:
    model = load_master_brain()
except Exception as e:
    st.error(f"Model error. Check path: {e}")
    st.stop()

# ─────────────────────────────────────────────
# 4. SIDEBAR (unchanged logic)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    plant_size = st.number_input("System Size (kW)", value=10.0, step=0.1)

    st.markdown("### 📍 Location")
    loc = get_geolocation()
    if loc:
        lat = loc['coords']['latitude']
        lon = loc['coords']['longitude']
        st.success("GPS Locked 📡")
    else:
        st.warning("Using manual coordinates")
        lat = st.number_input("Latitude",  value=51.5074)
        lon = st.number_input("Longitude", value=-0.1278)

    st.markdown("### 📐 Geometry")
    suggested_tilt = abs(lat)
    user_tilt    = st.slider("Panel Tilt",   0, 90,  int(suggested_tilt))
    user_azimuth = st.slider("Orientation",  0, 360, 180 if lat > 0 else 0)

# ─────────────────────────────────────────────
# 5. DATA ENGINE (unchanged)
# ─────────────────────────────────────────────
with st.spinner("Connecting to meteorological data..."):
    current_time  = pd.Timestamp.now(tz='UTC')
    start_of_day  = current_time.normalize()
    timeline      = pd.date_range(start=start_of_day, periods=96, freq='15min')
    solpos        = pvlib.solarposition.get_solarposition(timeline, lat, lon)

    w_url = (f"https://api.open-meteo.com/v1/forecast"
             f"?latitude={lat}&longitude={lon}"
             f"&minutely_15=cloud_cover&forecast_days=2"
             f"&hourly=weathercode,temperature_2m,windspeed_10m"
             f"&daily=sunrise,sunset"
             f"&current_weather=true"
             f"&timezone=auto")
    try:
        w_resp     = requests.get(w_url, timeout=10).json()
        weather_df = pd.DataFrame(w_resp['minutely_15'])
        weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
        weather_df = weather_df.set_index('time')

        # Current conditions
        cw           = w_resp.get('current_weather', {})
        cur_temp     = cw.get('temperature', 15)
        cur_wind     = cw.get('windspeed', 10)
        cur_wcode    = int(cw.get('weathercode', 0))
        is_day_api   = int(cw.get('is_day', 1))

        # Sunrise / sunset for today (index 0 = today)
        daily        = w_resp.get('daily', {})
        sunrise_str  = (daily.get('sunrise') or [''])[0]
        sunset_str   = (daily.get('sunset')  or [''])[0]
        try:
            sunrise_dt = pd.to_datetime(sunrise_str).to_pydatetime().replace(tzinfo=None)
            sunset_dt  = pd.to_datetime(sunset_str ).to_pydatetime().replace(tzinfo=None)
        except Exception:
            sunrise_dt = None
            sunset_dt  = None
    except Exception:
        weather_df = pd.DataFrame(index=timeline)
        weather_df['cloud_cover'] = 30
        cur_temp, cur_wind, cur_wcode, is_day_api = 15, 10, 0, 1
        sunrise_dt = None
        sunset_dt  = None

    prediction_rows = []
    for t in timeline:
        zenith   = solpos.loc[t, 'zenith']
        azimuth  = solpos.loc[t, 'azimuth']
        try:
            cloud_cover = weather_df.loc[t, 'cloud_cover']
        except KeyError:
            cloud_cover = 50
        aoi = pvlib.irradiance.aoi(user_tilt, user_azimuth, zenith, azimuth)
        prediction_rows.append({
            'datetime': t, 'CSI_trend_15min': 0.0,
            'zenith': float(zenith), 'azimuth': float(azimuth),
            'month': int(t.month), 'hour': int(t.hour),
            'lat': float(lat), 'lon': float(lon),
            'tilt': float(user_tilt), 'orientation': float(user_azimuth),
            'aoi': float(aoi), 'cloud_cover': float(cloud_cover)
        })

    full_day_df    = pd.DataFrame(prediction_rows)
    ai_features    = full_day_df.drop(columns=['datetime', 'aoi', 'cloud_cover'])
    dmatrix_input  = xgb.DMatrix(ai_features)
    ai_predictions = model.predict(dmatrix_input)

    final_power_list = []
    for idx, row in full_day_df.iterrows():
        if row['zenith'] > 90:
            final_power_list.append(0.0)
        else:
            angle_eff  = max(0.0, np.cos(np.radians(row['aoi'])))
            cloud_eff = max(0.08, (100 - row['cloud_cover']) / 100)
            pred_shift = ai_predictions[idx]
            watts      = (plant_size * 1000) * cloud_eff * angle_eff * (1 + pred_shift)
            final_power_list.append(max(0.0, watts))

    full_day_df['Predicted_Watts'] = final_power_list
    full_day_df['Power_kW']        = full_day_df['Predicted_Watts'] / 1000

# ─────────────────────────────────────────────
# 6. WEATHER CONDITION → BACKGROUND IMAGE
# ─────────────────────────────────────────────
# WMO weather codes → background scene
# Using high-quality Unsplash source images (no API key needed)
def get_weather_scene(wcode: int, is_day: int, cloud_pct: float):
    """Return (image_url, condition_label, overlay_opacity)"""

    # Night
    if not is_day:
        return (
            "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=1600&q=80&fit=crop",
            "Clear night", 0.60
        )
    # Thunderstorm (wcode 95-99)
    if wcode >= 95:
        return (
            "https://images.unsplash.com/photo-1605727216801-e27ce1d0cc28?w=1600&q=80&fit=crop",
            "Thunderstorm", 0.70
        )
    # Snow (wcode 71-77, 85-86)
    if wcode in range(71, 78) or wcode in (85, 86):
        return (
            "https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=1600&q=80&fit=crop",
            "Snowing", 0.58
        )
    # Rain / drizzle (wcode 51-67)
    if wcode in range(51, 68):
        return (
            "https://images.unsplash.com/photo-1519692933481-e162a57d6721?w=1600&q=80&fit=crop",
            "Rainy", 0.68
        )
    # Fog / mist (wcode 45-49)
    if wcode in (45, 48):
        return (
            "https://images.unsplash.com/photo-1482938289607-e9573fc25ebb?w=1600&q=80&fit=crop",
            "Foggy", 0.65
        )
    # Heavy cloud (wcode 3, or cloud > 75%)
    if wcode == 3 or cloud_pct > 75:
        return (
            "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=1600&q=80&fit=crop",
            "Overcast", 0.65
        )
    # Partly cloudy (wcode 2, or cloud 25-75%)
    if wcode == 2 or cloud_pct > 25:
        return (
            "https://images.unsplash.com/photo-1501630834273-4b5604d2ee31?w=1600&q=80&fit=crop",
            "Partly cloudy", 0.58
        )
    # Clear / sunny (wcode 0-1)
    return (
        "https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?w=1600&q=80&fit=crop",
        "Sunny", 0.50
    )

# Compute current values for scene selection
current_row    = full_day_df.iloc[(full_day_df['datetime'] - current_time).abs().argsort()[:1]].iloc[0]
current_kw     = current_row['Power_kW']
current_clouds = float(current_row['cloud_cover'])
peak_kw        = full_day_df['Power_kW'].max()
peak_idx       = full_day_df['Power_kW'].idxmax()
peak_time      = full_day_df.loc[peak_idx, 'datetime'].strftime('%H:%M')
cur_idx        = int((full_day_df['datetime'] - current_time).abs().argsort().iloc[0])
yield_so_far   = float(full_day_df.iloc[:cur_idx]['Power_kW'].sum() * 0.25)
total_yield    = float(full_day_df['Power_kW'].sum() * 0.25)
pct            = min(100, int(current_kw / plant_size * 100))

bg_url, condition_label, overlay_opacity = get_weather_scene(cur_wcode, is_day_api, current_clouds)

# ─────────────────────────────────────────────
# 6b. SUN ARC CALCULATIONS
# ─────────────────────────────────────────────
now_local = datetime.now()

if sunrise_dt and sunset_dt:
    day_secs    = (sunset_dt - sunrise_dt).total_seconds()
    elapsed     = (now_local - sunrise_dt).total_seconds()
    sun_pct     = max(0.0, min(1.0, elapsed / day_secs)) if day_secs > 0 else 0.5
    daylight_h  = day_secs / 3600
    sr_fmt      = sunrise_dt.strftime('%H:%M')
    ss_fmt      = sunset_dt.strftime('%H:%M')
else:
    sun_pct    = 0.5
    daylight_h = 14.0
    sr_fmt     = "06:00"
    ss_fmt     = "20:00"

def render_sun_arc(pct, sr, ss, dl_h, peak_t):
    """Renders the sun arc card using st.components.v1.html to avoid SVG/f-string escaping issues."""
    import math
    W, H = 340, 100
    cx, cy, r = W / 2, H, 80

    angle_rad = math.radians(180 - pct * 180)
    sx = cx + r * math.cos(angle_rad)
    sy = cy - r * math.sin(angle_rad)

    arc_len = math.pi * r
    filled  = round(pct * arc_len, 1)
    gap     = round(arc_len - filled, 1)
    bar_pct = int(pct * 100)

    html = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: 'DM Sans', -apple-system, sans-serif; }
  .card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 0.5px solid rgba(255,255,255,0.14);
    border-radius: 20px;
    padding: 16px 20px 14px;
  }
  .card-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.09em;
    color: rgba(255,255,255,0.38); margin-bottom: 10px;
  }
  .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
  .bar-track { flex: 1; height: 5px; background: rgba(255,255,255,0.12); border-radius: 10px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #ff9500, #ffd60a); }
  .sub { display: flex; justify-content: space-between; font-size: 10px; color: rgba(255,255,255,0.35); padding: 0 20px; }
</style>
<div class="card">
  <div class="card-label">Sun position</div>
  <svg viewBox="0 0 """ + str(W) + """ """ + str(H + 18) + """" xmlns="http://www.w3.org/2000/svg"
       style="width:100%;display:block;overflow:visible;margin-bottom:12px;">
    <defs>
      <radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
        <stop offset="0%"   stop-color="#ffd60a" stop-opacity="0.55"/>
        <stop offset="100%" stop-color="#ffd60a" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <path d="M """ + str(round(cx-r,1)) + """,""" + str(cy) + """ A """ + str(r) + """,""" + str(r) + """ 0 0,1 """ + str(round(cx+r,1)) + """,""" + str(cy) + """"
          fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="2" stroke-linecap="round"/>
    <path d="M """ + str(round(cx-r,1)) + """,""" + str(cy) + """ A """ + str(r) + """,""" + str(r) + """ 0 0,1 """ + str(round(cx+r,1)) + """,""" + str(cy) + """"
          fill="none" stroke="rgba(255,214,10,0.55)" stroke-width="2.5" stroke-linecap="round"
          stroke-dasharray=\"""" + str(filled) + " " + str(gap) + """\" />
    <circle cx=\"""" + str(round(sx,1)) + """\" cy=\"""" + str(round(sy,1)) + """\" r="22" fill="url(#sunGlow)"/>
    <circle cx=\"""" + str(round(sx,1)) + """\" cy=\"""" + str(round(sy,1)) + """\" r="10" fill="#ffd60a" opacity="0.95">
      <animate attributeName="r" values="10;12;10" dur="3s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.95;0.72;0.95" dur="3s" repeatCount="indefinite"/>
    </circle>
    <circle cx=\"""" + str(round(sx,1)) + """\" cy=\"""" + str(round(sy,1)) + """\" r="5" fill="#fff" opacity="0.58"/>
    <line x1=\"""" + str(round(cx-r-6,1)) + """\" y1=\"""" + str(cy) + """\" x2=\"""" + str(round(cx+r+6,1)) + """\" y2=\"""" + str(cy) + """"
          stroke="rgba(255,255,255,0.14)" stroke-width="1"/>
    <text x=\"""" + str(round(cx-r,1)) + """\" y=\"""" + str(cy+15) + """\" font-size="10" fill="rgba(255,255,255,0.45)"
          text-anchor="middle" font-family="DM Sans,-apple-system,sans-serif">""" + sr + """</text>
    <text x=\"""" + str(round(cx+r,1)) + """\" y=\"""" + str(cy+15) + """\" font-size="10" fill="rgba(255,255,255,0.45)"
          text-anchor="middle" font-family="DM Sans,-apple-system,sans-serif">""" + ss + """</text>
    <text x=\"""" + str(round(cx,1)) + """\" y=\"""" + str(cy+15) + """\" font-size="10" fill="rgba(255,255,255,0.38)"
          text-anchor="middle" font-family="DM Sans,-apple-system,sans-serif">""" + str(round(dl_h,1)) + """h daylight</text>
  </svg>
  <div class="bar-row">
    <span style="font-size:13px;">🌅</span>
    <div class="bar-track">
      <div class="bar-fill" style="width:""" + str(bar_pct) + """%;"></div>
    </div>
    <span style="font-size:13px;">🌇</span>
  </div>
  <div class="sub">
    <span>""" + sr + """</span>
    <span>Peak """ + peak_t + """</span>
    <span>""" + ss + """</span>
  </div>
</div>
"""
    components.html(html, height=230, scrolling=False)

# ─────────────────────────────────────────────
# 7. CSS — background image + full theme
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

/* ── LIVE WEATHER BACKGROUND ── */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image: url('{bg_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    filter: blur(3px) brightness(0.75) saturate(0.9);
    transform: scale(1.05);
    transition: background-image 1.5s ease-in-out;
    animation: bgPan 60s ease-in-out infinite alternate;
}}
@keyframes bgPan {{
    0%   {{ background-position: center 40%; }}
    100% {{ background-position: center 60%; }}
}}

/* ── Dark gradient overlay on top of photo ── */
[data-testid="stAppViewContainer"]::after {{
    content: '';
    position: fixed;
    inset: 0;
    z-index: 1;
    background:
        linear-gradient(160deg,
            rgba(5,12,28,{overlay_opacity + 0.10}) 0%,
            rgba(5,12,28,{overlay_opacity})       50%,
            rgba(5,12,28,{overlay_opacity + 0.05}) 100%);
    pointer-events: none;
}}

[data-testid="stAppViewContainer"] {{
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    position: relative;
}}

/* Push all Streamlit content above the pseudo-elements */
[data-testid="stAppViewContainer"] > * {{ position: relative; z-index: 2; }}
[data-testid="stHeader"] {{ background: transparent !important; z-index: 10 !important; }}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: rgba(5,12,28,0.80) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border-right: 0.5px solid rgba(255,255,255,0.08) !important;
    z-index: 10 !important;
}}
[data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.80) !important; }}
[data-testid="stSidebarContent"] {{ padding-top: 2rem; }}

/* ── LAYOUT ── */
.block-container {{ padding: 2rem 2rem 1rem !important; max-width: 1200px !important; }}
html, body, [class*="css"] {{ color: #ffffff !important; }}

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {{
    background:  rgba(255,255,255,0.10) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 0.5px solid rgba(255,255,255,0.18) !important;
    border-radius: 18px !important;
    padding: 20px 22px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.20) !important;
    transition: background 0.2s;
}}
[data-testid="metric-container"]:hover {{
    background: rgba(255,255,255,0.14) !important;
}}
[data-testid="stMetricLabel"] {{
    color: rgba(255,255,255,0.50) !important;
    font-size: 0.70rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 400 !important;
}}
[data-testid="stMetricValue"] {{
    color: #ffffff !important;
    font-size: 1.85rem !important;
    font-weight: 300 !important;
    letter-spacing: -0.5px;
}}
[data-testid="stMetricDelta"] {{ color: rgba(255,255,255,0.55) !important; }}

hr {{ border-color: rgba(255,255,255,0.08) !important; margin: 1.2rem 0 !important; }}
h1,h2,h3 {{ color: #ffffff !important; font-weight: 400 !important; }}
[data-testid="stCaptionContainer"] p {{ color: rgba(255,255,255,0.38) !important; font-size: 0.78rem !important; }}
[data-testid="stSpinner"]            {{ color: rgba(255,255,255,0.50) !important; }}
[data-testid="stNumberInput"] input  {{
    background: rgba(255,255,255,0.06) !important;
    border: 0.5px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important; color: white !important;
}}
::-webkit-scrollbar       {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 10px; }}

/* pulse animation */
@keyframes pulse {{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.8)}}}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. DASHBOARD UI
# ─────────────────────────────────────────────

# ── HEADER ───────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:0.5rem;">
  <div>
    <div style="font-size:13px; color:rgba(255,255,255,0.50); letter-spacing:0.04em; margin-bottom:4px;">
      <span style="display:inline-block; width:7px; height:7px; border-radius:50%;
             background:#30d158; box-shadow:0 0 7px #30d158;
             margin-right:6px; vertical-align:middle;"></span>
      {lat:.4f}°N, {lon:.4f}°E
    </div>
    <div style="font-size:26px; font-weight:500; letter-spacing:-0.5px; color:#fff;
                text-shadow:0 2px 12px rgba(0,0,0,0.40);">Solar Forecast</div>
    <div style="font-size:12px; color:rgba(255,255,255,0.35); margin-top:2px;">
      Updated {datetime.now().strftime('%H:%M:%S')} &nbsp;·&nbsp; Auto-refreshing every 5 min
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:8px; padding-top:4px;">
    <div style="background:rgba(48,209,88,0.15); border:0.5px solid rgba(48,209,88,0.35);
                border-radius:20px; padding:6px 14px; font-size:13px; color:#30d158;
                display:flex; align-items:center; gap:6px;
                backdrop-filter:blur(12px);">
      <span style="width:7px;height:7px;border-radius:50%;background:#30d158;
                   animation:pulse 2s ease-in-out infinite;display:inline-block;"></span>
      Live
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── HERO + SUN ARC ───────────────────────────
hero_col, arc_col = st.columns([3, 2])

with hero_col:
    st.markdown(f"""
    <div style="display:flex; align-items:flex-end; justify-content:space-between;
                margin:1.4rem 0 1.2rem; gap:16px;">
      <div>
        <div style="font-size:82px; font-weight:300; letter-spacing:-3px; line-height:1;
                    color:#fff; text-shadow:0 4px 24px rgba(0,0,0,0.50);">
          {current_kw:.1f}<span style="font-size:32px; font-weight:400;
            color:rgba(255,255,255,0.55); margin-left:4px; letter-spacing:0;">kW</span>
        </div>
        <div style="font-size:14px; color:rgba(255,255,255,0.60); margin-top:8px;
                    display:flex; align-items:center; gap:12px;">
          <span>{pct}% of {plant_size:.0f} kW system</span>
          <span style="background:rgba(255,214,10,0.15); border:0.5px solid rgba(255,214,10,0.32);
                       border-radius:8px; padding:3px 10px; font-size:12px; color:#ffd60a;
                       backdrop-filter:blur(8px);">⚡ Generating now</span>
        </div>
      </div>
      <div style="text-align:right; background:rgba(0,0,0,0.25); backdrop-filter:blur(16px);
                  border-radius:16px; padding:14px 18px;
                  border:0.5px solid rgba(255,255,255,0.12);">
        <div style="font-size:11px; color:rgba(255,255,255,0.45); margin-bottom:4px;
                    text-transform:uppercase; letter-spacing:0.08em;">Live conditions</div>
        <div style="font-size:18px; color:#fff; font-weight:400;">{condition_label}</div>
        <div style="font-size:13px; color:rgba(255,255,255,0.50); margin-top:4px;">
          {cur_temp:.0f}°C &nbsp;·&nbsp; {cur_wind:.0f} km/h &nbsp;·&nbsp; ☁️ {int(current_clouds)}%
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with arc_col:
    st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)
    render_sun_arc(sun_pct, sr_fmt, ss_fmt, daylight_h, peak_time)

# ── METRICS ──────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("TODAY'S PEAK",    f"{peak_kw:.1f} kW",      f"At {peak_time}")
c2.metric("YIELD SO FAR",    f"{yield_so_far:.1f} kWh", f"Est. {total_yield:.0f} kWh total")
c3.metric("CLOUD COVER",     f"{int(current_clouds)}%", condition_label)
c4.metric("CAPACITY FACTOR", f"{pct}%",                f"{plant_size:.0f} kW · {user_tilt}° tilt")

st.markdown("<div style='margin:1.4rem 0 0.4rem;'></div>", unsafe_allow_html=True)

# ── 24-HOUR CHART ─────────────────────────────
st.markdown("""
<div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
            color:rgba(255,255,255,0.38); margin-bottom:10px;">
  24-hour power forecast
</div>""", unsafe_allow_html=True)

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(
    x=full_day_df['datetime'], y=full_day_df['Power_kW'],
    name="Power (kW)",
    fill='tozeroy', mode='lines', line_shape='spline',
    line=dict(color='#ffd60a', width=2.5),
    fillcolor='rgba(255,214,10,0.18)'
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=full_day_df['datetime'], y=full_day_df['cloud_cover'],
    name="Cloud (%)",
    fill='tozeroy', mode='lines', line_shape='spline',
    line=dict(color='rgba(100,210,255,0.65)', width=1.5, dash='dot'),
    fillcolor='rgba(100,210,255,0.07)'
), secondary_y=True)

fig.add_vline(
    x=current_time.tz_localize(None),
    line_width=1.5, line_dash="dash",
    line_color="rgba(255,255,255,0.30)"
)

fig.update_layout(
    height=220,
    hovermode="x unified",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=8, b=0),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color="rgba(255,255,255,0.50)", size=11),
        bgcolor="rgba(0,0,0,0)"
    ),
    xaxis=dict(
        showgrid=True, gridcolor='rgba(255,255,255,0.06)',
        tickfont=dict(color="rgba(255,255,255,0.35)", size=10),
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True, gridcolor='rgba(255,255,255,0.06)',
        tickfont=dict(color="rgba(255,255,255,0.35)", size=10),
        zeroline=False
    ),
    yaxis2=dict(
        tickfont=dict(color="rgba(100,210,255,0.45)", size=10),
        zeroline=False
    )
)
fig.update_yaxes(title_text="", secondary_y=False, range=[0, plant_size])
fig.update_yaxes(title_text="", secondary_y=True,  range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

# ── HOURLY RIBBON ────────────────────────────
st.markdown("""
<div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
            color:rgba(255,255,255,0.38); margin:1.2rem 0 10px;">
  Next 8 hours
</div>""", unsafe_allow_html=True)

future_data = full_day_df[full_day_df['datetime'] >= current_time.floor('h')]
hourly_data = future_data.iloc[::4].head(8)

if not hourly_data.empty:
    cols = st.columns(8)
    for i, (idx, row) in enumerate(hourly_data.iterrows()):
        if i >= 8: break
        hour_str = "Now" if i == 0 else row['datetime'].strftime("%H:%M")
        kw_val   = row['Power_kW']
        cld_val  = int(row['cloud_cover'])
        is_now   = i == 0

        if row['zenith'] > 90:     icon = "🌙"
        elif cld_val > 75:         icon = "🌧️" if cld_val > 90 else "☁️"
        elif cld_val > 30:         icon = "⛅"
        else:                      icon = "☀️"

        border = ("0.5px solid rgba(48,209,88,0.55)" if is_now
                  else "0.5px solid rgba(255,255,255,0.12)")
        bg     = ("rgba(48,209,88,0.12)" if is_now
                  else "rgba(255,255,255,0.08)")

        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center; background:{bg};
                        backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
                        padding:14px 6px; border-radius:16px; border:{border};
                        transition:background 0.2s; box-shadow:0 4px 16px rgba(0,0,0,0.20);">
              <div style="font-size:10px; color:rgba(255,255,255,0.40);
                          margin-bottom:7px; font-variant-numeric:tabular-nums;">{hour_str}</div>
              <div style="font-size:22px; margin-bottom:6px;">{icon}</div>
              <div style="font-size:14px; font-weight:500; color:#fff;">{kw_val:.1f} kW</div>
              <div style="font-size:10px; color:rgba(255,255,255,0.40);
                          margin-top:3px;">{cld_val}% cloud</div>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────
st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:rgba(0,0,0,0.30); border:0.5px solid rgba(255,255,255,0.10);
            backdrop-filter:blur(20px); border-radius:16px; padding:12px 20px;
            display:flex; align-items:center; gap:20px;
            font-size:11px; color:rgba(255,255,255,0.35);">
  <span>🌅 {condition_label}</span>
  <span style="width:3px;height:3px;border-radius:50%;
               background:rgba(255,255,255,0.25);display:inline-block;"></span>
  <span>System: {plant_size:.0f} kW</span>
  <span style="width:3px;height:3px;border-radius:50%;
               background:rgba(255,255,255,0.25);display:inline-block;"></span>
  <span>Tilt: {user_tilt}° &nbsp;·&nbsp; Azimuth: {user_azimuth}°</span>
  <span style="width:3px;height:3px;border-radius:50%;
               background:rgba(255,255,255,0.25);display:inline-block;"></span>
  <span>Open-Meteo · pvlib · XGBoost</span>
  <span style="margin-left:auto;">{datetime.now().strftime('%A, %d %b %Y')}</span>
</div>
""", unsafe_allow_html=True)