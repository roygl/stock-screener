"""Stock screener — Streamlit entry point.

Milestone 1 skeleton: confirms the app boots and the ticker universe loads.
Profiles, indicators, and ranking arrive in later milestones; the controls
below are intentionally inert placeholders so the layout is visible.

Run locally:
    streamlit run app.py
"""

import streamlit as st

from screener.universe import load_universe

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")

st.title("📈 Stock Screener")
st.caption("US large-cap equities · end-of-day · ranks and describes, never advises")

# --- Load the universe ---------------------------------------------------
try:
    universe = load_universe()
except Exception as exc:  # surface load failures in the UI instead of a stack trace
    st.error(f"Could not load the ticker universe: {exc}")
    st.stop()

n_tickers = len(universe)
n_sectors = universe["sector"].nunique()

col_a, col_b = st.columns(2)
col_a.metric("Tickers in universe", f"{n_tickers}")
col_b.metric("Sectors", f"{n_sectors}")

# --- Inert placeholders (wired up in later milestones) -------------------
with st.sidebar:
    st.header("Controls")
    st.selectbox(
        "Profile",
        ["Long-term", "Swing", "Momentum/Growth"],
        help="Ranking profile — connected to the engine in Milestone 4.",
        disabled=True,
    )
    st.selectbox(
        "Asset class",
        ["US equities"],
        help="Crypto arrives in v2.",
        disabled=True,
    )
    st.info("Profiles and ranking are added in later milestones.")

# --- Universe preview ----------------------------------------------------
st.subheader("Universe")
st.caption("Loaded from the static `data/universe.csv` file.")

sectors = ["All"] + sorted(universe["sector"].unique())
chosen = st.selectbox("Filter by sector", sectors)
view = universe if chosen == "All" else universe[universe["sector"] == chosen]

st.dataframe(
    view.rename(columns={"symbol": "Symbol", "name": "Name", "sector": "Sector"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Showing {len(view)} of {n_tickers} tickers.")
