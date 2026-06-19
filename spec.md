# spec.md — Stock Screener Specification

_Last updated: 2026-06-19 · Status: stable (v1 scope locked)_

---

## 1. User
An individual investor screening **US large-cap equities** to find candidates that
fit a chosen **style/horizon** — from long-term holdings to short-term swing trades.
The user already trades manually with FINVIZ / TradingView / a broker; this tool
**automates the systematic scanning and watchlist-building step**, it does not
replace charting, alerts, or execution.

## 2. Problem
Scanning the market for names that fit a specific strategy is slow and scattered
across many sites. The numbers that matter differ completely by style — a long-term
investor cares about valuation and growth; a swing trader cares about relative volume,
breakouts, and moving-average structure. This tool ranks one universe through the
**lens of the style you select**, so the weekend scan becomes one click.

## 3. What it is
A **web dashboard** (Streamlit). Pick a profile → see a ranked, sortable, filterable
table of matching tickers, with a per-row "why it ranks" breakdown. A natural-language
**agent layer** that translates plain English into screen parameters is a *later*
addition, not part of v1.

## 4. Scope
| Area | v1 (MVP) | Later |
|---|---|---|
| Asset class | US large-cap equities | Crypto (v2), other equities |
| Interface | Dashboard | Agent / chat layer |
| Data freshness | End-of-day | Intraday / real-time (paid API) |
| Profiles | Long-term, Swing, Momentum/Growth | Value/Income, user-saved custom |

## 5. Profiles
Each profile is a **config**: which variables to filter on, and how to weight them
for ranking. Same engine, different lens. Moving averages are **per-profile**, not global.

- **Long-term:** valuation (forward P/E), revenue/earnings growth, trend structure
  (price above 20/50/150 SMA, stacked 20 > 50 > 150), distance from 52-wk high.
- **Swing:** relative volume > 2×, 5/9 EMA cross, MACD/RSI, 10/20 EMA pullback,
  leading-sector membership, earnings-window flag.
- **Momentum/Growth:** 1/3/6/12-mo momentum, 20/50/150 SMA structure, relative volume,
  distance from 52-wk high, growth.

> New variables get added to a profile config, not hard-coded into the engine.

## 6. Variables
momentum (1/3/6/12-mo), forward P/E, RSI, MACD, relative volume, 52-wk-high distance,
revenue/earnings growth, market cap, sector, plus profile-specific MAs —
**20/50/150 SMA** for long-term & momentum, **5/9 EMA** for swing.

## 7. Data sources
- **Equities (v1):** `yfinance` — free, end-of-day; price history (for momentum,
  MAs, RSI, MACD, relative volume) and fundamentals (forward P/E, growth, market cap,
  sector, earnings date). Unofficial/scraping-based, so validate and cache.
- **Upgrade path:** a paid structured API (e.g. Financial Modeling Prep) for reliable
  forward estimates and intraday — swapped in behind the `DataProvider` interface.
- **Crypto (v2):** CoinGecko free Demo tier (~100 calls/min, 10k/month).
- **Catalysts (stretch):** SEC EDGAR (sec.gov) for authoritative filings — 8-K
  (material events), Form 4 (insider trades), 13F (institutional holdings).

## 8. Success criteria
- Pick a profile → a ranked list returns within a few seconds.
- Numbers spot-check correctly against Yahoo Finance (and CoinGecko in v2).
- The table sorts and filters, and each row shows **why** it ranks where it does.
- The swing profile correctly surfaces high-relative-volume breakout candidates in a
  leading sector and flags any with imminent earnings.

## 9. Non-goals (v1)
- No buy / sell calls or personalized advice — it **describes and ranks**, never recommends.
- No backtesting, no portfolio tracking, no order execution.
- No intraday / real-time data (end-of-day only).
- No user accounts, no custom saved profiles.
- No charting or live alerting (that stays in TradingView / the broker).
- No agent / chat layer.

## 10. Philosophy & disclaimers (carried from the user's own notes)
- A hard stop-loss belongs on every swing position; position sizing and stops protect
  more than any tool — the screener finds candidates, it does not manage risk.
- Don't chase names already overextended from their moving average.
- Most active traders underperform buy-and-hold over time; paper-trade first, size
  small while learning.
- This tool is **not financial advice** and its author is not a licensed advisor.

## 11. Open questions
- Hosting: run locally vs deploy (Streamlit Community Cloud)?
- Universe definition for "US large-cap": S&P 500? Russell 1000? A market-cap floor?
  _(M1 resolved the starting point: static S&P 500 list.)_
- Refresh cadence for the end-of-day data pull?
