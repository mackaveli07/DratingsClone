# ui_helpers.py
import base64
import html
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LOGOS_DIR = APP_DIR / "Logos"
SHIELD_IMAGE_PATH = APP_DIR / "Shield.png"
NFL_IMAGE_PATH = APP_DIR / "NFL.png"

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LV": "#000000", "LAC": "#0080C6", "LA": "#003594", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SF": "#AA0000",
    "SEA": "#002244", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414"
}


def safe_logo(abbr, width=64):
    path = LOGOS_DIR / f"{abbr}.png"
    safe_abbr = html.escape(str(abbr or "?"))
    if abbr and path.exists():
        try:
            st.image(str(path), width=width)
        except Exception:
            st.markdown(
                f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
                f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
                f"font-size:12px; color:#475569;'>{safe_abbr}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
            f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
            f"font-size:12px; color:#475569;'>{safe_abbr}</div>",
            unsafe_allow_html=True,
        )


def neon_text(text, abbr=None, size=24):
    color = TEAM_COLORS.get(abbr, "#39ff14") if abbr else "#39ff14"
    safe_text = html.escape(str(text))
    return f"""
    <span style="
        color: #f8fafc;
        font-size: {size}px;
        font-weight: 800;
        letter-spacing: 0.02em;
        line-height: 1.15;
        -webkit-text-stroke: 1px {color};
        text-shadow:
            0 0 2px rgba(15, 23, 42, 0.95),
            0 0 8px {color},
            0 0 18px {color},
            0 0 30px {color};
    ">{safe_text}</span>
    """


def set_background(image_path=SHIELD_IMAGE_PATH):
    image_path = Path(image_path)
    if image_path.exists():
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)),
                            url("data:image/png;base64,{b64}") no-repeat center center fixed;
                background-size: cover;
                color: white;
            }}
            /* Frosted glass cards */
            .card {{
                background: rgba(30,30,30,0.6);
                backdrop-filter: blur(14px);
                border-radius: 20px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            }}
            @media (max-width: 768px) {{
                h1,h2,h3,h4,h5,h6 {{ font-size:90% !important; }}
                .card {{ padding:14px !important; margin:12px 0 !important; }}
                img {{ max-width:80px !important; height:auto !important; }}
                .stMarkdown p {{ font-size:14px !important; }}
            }}
            @media (max-width: 480px) {{
                .card {{ padding:10px !important; }}
                h1,h2,h3 {{ font-size:80% !important; }}
                img {{ max-width:60px !important; }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


set_background()


def load_local_logo(path=NFL_IMAGE_PATH):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


NFL_LOGO_B64 = load_local_logo()


def nfl_header(title):
    safe_title = html.escape(str(title))
    logo_html = f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='60'>" if NFL_LOGO_B64 else ""
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #013369, #d50a0a); 
                    padding: 20px; border-radius: 15px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:16px;'>""" + logo_html + f"""
            <h1 style='color:white; margin:0; font-size:42px;'>{safe_title}</h1>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def nfl_subheader(text, icon="📊"):
    safe_text = html.escape(str(text))
    safe_icon = html.escape(str(icon))
    logo_html = f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='32' style='margin-right:8px;'/>" if NFL_LOGO_B64 else ""
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #d50a0a, #013369); 
                    padding: 12px; border-radius: 12px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:10px;'>""" + logo_html + f"""
            <h2 style='color:white; margin:0;'>{safe_icon} {safe_text}</h2>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
