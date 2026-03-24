#!/usr/bin/env python3
"""
Squeeze Scanner — Live Terminal Dashboard

Reads data from VPS SQLite (single SSH call) + Bybit API for BTC price.
Shows 5 parameter-ranked tables from latest ml_snapshots + recent events.
Refreshes every 60 seconds. Ctrl+C to quit.

Usage:
    python terminal_dashboard.py
    VPS_PASSWORD=xxx python terminal_dashboard.py

Requirements:
    pip install rich paramiko requests
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import paramiko
import requests
from rich import box
from rich.console import Console
from rich.columns import Columns
from rich.table import Table
from rich.text import Text

# ── Config ────────────────────────────────────────────────────────────────────

VPS_HOST = "204.168.158.206"
VPS_USER = "root"
VPS_DB = "/root/bybit-perp-scanner/data/events.db"
REFRESH_S = 60

# ── SSH ───────────────────────────────────────────────────────────────────────

_ssh_client: Optional[paramiko.SSHClient] = None


def _get_ssh() -> paramiko.SSHClient:
    global _ssh_client
    if _ssh_client is not None:
        try:
            _ssh_client.exec_command("echo ok", timeout=5)
            return _ssh_client
        except Exception:
            try:
                _ssh_client.close()
            except Exception:
                pass
            _ssh_client = None

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pw = os.environ.get("VPS_PASSWORD")
    kwargs: Dict[str, Any] = {"hostname": VPS_HOST, "username": VPS_USER, "timeout": 10}
    if pw:
        kwargs["password"] = pw
    client.connect(**kwargs)
    _ssh_client = client
    return client


def ssh_exec(cmd: str) -> str:
    client = _get_ssh()
    _, stdout, _ = client.exec_command(cmd, timeout=20)
    return stdout.read().decode().strip()


# ── Data fetching (ONE SSH call) ─────────────────────────────────────────────

_MEGA_QUERY = r"""
echo '===SNAP===';
sqlite3 -separator '|' DB "
SELECT symbol, funding_rate, oi_change_1h_pct, oi_change_4h_pct,
       thin_pct, bb_width_pct, vacuum_dist_ask, vacuum_dist_bid,
       oi_usd, depth_ask_usdt, depth_bid_usdt
FROM ml_snapshots
WHERE timestamp = (SELECT MAX(timestamp) FROM ml_snapshots)
  AND mid_price IS NOT NULL
ORDER BY symbol";

echo '===EVENTS===';
sqlite3 -separator '|' DB "
SELECT timestamp, event_type, symbol, score, direction
FROM events
WHERE timestamp > datetime('now','-2 hours')
ORDER BY timestamp DESC
LIMIT 10";

echo '===EVTSUM===';
sqlite3 -separator '|' DB "
SELECT symbol, COUNT(DISTINCT event_type), MAX(score)
FROM events
WHERE timestamp > datetime('now','-2 hours') AND score >= 50
GROUP BY symbol";

echo '===STATS===';
sqlite3 -separator '|' DB "
SELECT
  (SELECT COUNT(*) FROM events WHERE timestamp > datetime('now','-24 hours')),
  (SELECT COUNT(*) FROM ml_snapshots),
  (SELECT SUM(label_filled) FROM ml_snapshots),
  (SELECT COUNT(*) FROM outcomes)";

echo '===MEM===';
free -m 2>/dev/null | awk 'NR==2{printf "%d/%dMB (%.0f%%)", $3,$2,$3*100/$2}';
""".replace("DB", VPS_DB)


def _f(s: str) -> float:
    try:
        return float(s.strip()) if s.strip() else 0.0
    except (ValueError, TypeError):
        return 0.0


def fetch_all() -> Dict[str, Any]:
    raw = ssh_exec(_MEGA_QUERY)

    sections: Dict[str, str] = {}
    current_key = ""
    current_lines: List[str] = []
    for line in raw.split("\n"):
        if line.startswith("===") and line.endswith("==="):
            if current_key:
                sections[current_key] = "\n".join(current_lines)
            current_key = line.strip("=")
            current_lines = []
        else:
            current_lines.append(line)
    if current_key:
        sections[current_key] = "\n".join(current_lines)

    # Parse all snapshots
    snaps: List[Dict[str, Any]] = []
    for line in sections.get("SNAP", "").split("\n"):
        if not line.strip():
            continue
        p = line.split("|")
        if len(p) >= 11:
            snaps.append({
                "symbol": p[0],
                "funding_rate": _f(p[1]),
                "oi_change_1h": _f(p[2]),
                "oi_change_4h": _f(p[3]),
                "thin_pct": _f(p[4]),
                "bb_width": _f(p[5]),
                "vac_ask": _f(p[6]),
                "vac_bid": _f(p[7]),
                "oi_usd": _f(p[8]),
                "depth_ask": _f(p[9]),
                "depth_bid": _f(p[10]),
            })

    # Event summary per symbol (for score column)
    evt_sum: Dict[str, Tuple[int, float]] = {}  # symbol -> (types, max_score)
    for line in sections.get("EVTSUM", "").split("\n"):
        if not line.strip():
            continue
        p = line.split("|")
        if len(p) >= 3:
            evt_sum[p[0]] = (int(_f(p[1])), _f(p[2]))

    # Recent events
    events: List[Dict[str, Any]] = []
    for line in sections.get("EVENTS", "").split("\n"):
        if not line.strip():
            continue
        p = line.split("|")
        if len(p) >= 5:
            events.append({
                "timestamp": p[0],
                "event_type": p[1],
                "symbol": p[2],
                "score": _f(p[3]),
                "direction": p[4],
            })

    # Stats
    stats: Dict[str, Any] = {}
    st = sections.get("STATS", "").strip()
    if st:
        p = st.split("|")
        if len(p) >= 4:
            stats["events_24h"] = int(_f(p[0]))
            stats["snapshots"] = int(_f(p[1]))
            stats["labeled"] = int(_f(p[2]))
            stats["outcomes"] = int(_f(p[3]))
    stats["memory"] = sections.get("MEM", "").strip() or "N/A"

    return {"snaps": snaps, "evt_sum": evt_sum, "events": events, "stats": stats}


# ── Bybit API ────────────────────────────────────────────────────────────────


def get_btc() -> Dict[str, float]:
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=5,
        )
        d = r.json()["result"]["list"][0]
        return {"price": float(d["lastPrice"]), "change_24h": float(d["price24hPcnt"]) * 100}
    except Exception:
        return {"price": 0, "change_24h": 0}


# ── Settlement ────────────────────────────────────────────────────────────────


def next_settlement() -> Tuple[int, int]:
    now = datetime.now(timezone.utc)
    cur = now.hour * 60 + now.minute
    for h in (0, 8, 16, 24):
        if h * 60 > cur:
            return h % 24, h * 60 - cur
    return 0, 1440 - cur


# ── Table builders ────────────────────────────────────────────────────────────

_COLS = [
    ("Symbol", "bold", 13, "left"),
    ("Value", "bold cyan", 10, "right"),
    ("FR%", None, 8, "right"),
    ("OI\u03941h", None, 7, "right"),
    ("Thin", None, 5, "right"),
    ("BB%", None, 5, "right"),
    ("Scr", None, 4, "right"),
]


def _make_table(
    title: str,
    title_style: str,
    rows: List[Dict[str, Any]],
    value_fn,
    value_fmt,
    evt_sum: Dict[str, Tuple[int, float]],
) -> Table:
    t = Table(title=title, box=box.SIMPLE_HEAVY, title_style=title_style,
              show_lines=False, pad_edge=False, expand=False)
    for name, style, width, just in _COLS:
        t.add_column(name, style=style, width=width, justify=just)

    for r in rows[:5]:
        fr = r["funding_rate"]
        fr_s = "red" if fr < -0.001 else "green" if fr > 0.001 else "white"
        thin_s = "red" if r["thin_pct"] > 0.90 else "yellow" if r["thin_pct"] > 0.80 else "white"
        oi_s = "yellow" if abs(r["oi_change_1h"]) > 30 else "white"
        _, score = evt_sum.get(r["symbol"], (0, 0))
        score_s = f"{score:.0f}" if score else "[dim]-[/]"

        t.add_row(
            r["symbol"],
            value_fmt(r),
            f"[{fr_s}]{fr*100:+.3f}%[/]",
            f"[{oi_s}]{r['oi_change_1h']:+.0f}%[/]",
            f"[{thin_s}]{r['thin_pct']:.2f}[/]",
            f"{r['bb_width']:.1f}",
            score_s,
        )

    if not rows:
        t.add_row("[dim]None[/]", *[""] * 6)
    return t


# ── Rendering ─────────────────────────────────────────────────────────────────

_EVENT_LABEL = {
    "CASCADE_ACTIVE": ("\U0001f534 CASCADE", "bold red"),
    "VOLUME_EXPLOSION": ("\u26a1 VOL_EXPL", "bold yellow"),
    "OI_SURGE": ("\U0001f4c8 OI_SURGE", "yellow"),
    "FUNDING_SQUEEZE_SETUP": ("\U0001f4b0 FUNDING", "white"),
    "VACUUM_BREAK": ("\U0001f300 VACUUM", "dim"),
    "COMPRESSION_SQUEEZE_SETUP": ("\U0001f5dc CS_SETUP", "dim"),
}


def render(console: Console, data: Dict[str, Any], btc: Dict[str, float]) -> None:
    now = datetime.now(timezone.utc)
    s_hour, s_mins = next_settlement()
    snaps = data["snaps"]
    evt_sum = data["evt_sum"]
    events = data["events"]
    stats = data["stats"]

    console.clear()

    # ── Header ────────────────────────────────────────────────────────────
    btc_c = btc["change_24h"]
    btc_dot = "\U0001f7e2" if btc_c > 1 else "\U0001f534" if btc_c < -1 else "\u27a1\ufe0f"
    settle_style = "bold red" if s_mins <= 30 else "bold yellow" if s_mins <= 60 else "cyan"

    hdr = Text()
    hdr.append(" SQUEEZE SCANNER ", style="bold white on blue")
    hdr.append(f"  BTC ${btc['price']:,.0f} {btc_c:+.1f}% {btc_dot}", style="bold")
    hdr.append(f"  |  {now:%H:%M:%S} UTC", style="dim")
    hdr.append(f"  |  Settlement {s_hour:02d}:00 in ", style="dim")
    hdr.append(f"{s_mins}min", style=settle_style)
    hdr.append(f"  |  {len(snaps)} coins", style="dim")
    console.print(hdr)
    console.print()

    # ── 5 parameter tables ────────────────────────────────────────────────

    # 1. Funding extremes
    by_funding = sorted(snaps, key=lambda s: abs(s["funding_rate"]), reverse=True)
    t_fund = _make_table(
        "\U0001f4b0 FUNDING EXTREMES", "bold red",
        by_funding, lambda s: abs(s["funding_rate"]),
        lambda s: f"{s['funding_rate']*100:+.4f}%",
        evt_sum,
    )

    # 2. OI acceleration
    by_oi = sorted(snaps, key=lambda s: abs(s["oi_change_1h"]), reverse=True)
    t_oi = _make_table(
        "\U0001f4c8 OI ACCELERATION", "bold yellow",
        by_oi, lambda s: abs(s["oi_change_1h"]),
        lambda s: f"{s['oi_change_1h']:+.1f}%",
        evt_sum,
    )

    # 3. Compression (lowest BB width, exclude 0/NULL)
    by_bb = sorted(
        [s for s in snaps if s["bb_width"] > 0],
        key=lambda s: s["bb_width"],
    )
    t_bb = _make_table(
        "\U0001f5dc COMPRESSION (low BB)", "bold magenta",
        by_bb, lambda s: s["bb_width"],
        lambda s: f"{s['bb_width']:.2f}%",
        evt_sum,
    )

    # 4. Thinnest books
    by_thin = sorted(snaps, key=lambda s: s["thin_pct"], reverse=True)
    t_thin = _make_table(
        "\U0001f4d6 THINNEST BOOKS", "bold cyan",
        by_thin, lambda s: s["thin_pct"],
        lambda s: f"{s['thin_pct']:.3f}",
        evt_sum,
    )

    # 5. Empty vacuum
    vacuums = [s for s in snaps if s["vac_ask"] > 5000 or s["vac_bid"] > 5000]
    vacuums.sort(key=lambda s: max(s["vac_ask"], s["vac_bid"]), reverse=True)
    t_vac = _make_table(
        "\U0001f300 EMPTY VACUUM", "bold white",
        vacuums, lambda s: max(s["vac_ask"], s["vac_bid"]),
        lambda s: f"{max(s['vac_ask'], s['vac_bid']):.0f} bps",
        evt_sum,
    )

    # Print tables in rows of 2 + 1
    console.print(Columns([t_fund, t_oi], padding=(0, 2)))
    console.print()
    console.print(Columns([t_bb, t_thin], padding=(0, 2)))
    console.print()
    console.print(t_vac)
    console.print()

    # ── Recent Events ─────────────────────────────────────────────────────
    t_ev = Table(
        title="RECENT EVENTS (last 2h)",
        box=box.SIMPLE_HEAVY,
        title_style="bold green",
    )
    t_ev.add_column("Time", width=6)
    t_ev.add_column("Type", width=18)
    t_ev.add_column("Symbol", width=14)
    t_ev.add_column("Score", justify="right", width=6)
    t_ev.add_column("Dir", width=4)

    for ev in events[:8]:
        ts = ev["timestamp"]
        try:
            time_str = ts.split("T")[1][:5] if "T" in ts else ts[11:16]
        except Exception:
            time_str = ts[:5]
        et = ev["event_type"]
        label, style = _EVENT_LABEL.get(et, (et[:12], "white"))
        dir_dot = "\U0001f7e2" if ev["direction"] in ("LONG", "UP") else "\U0001f534"
        t_ev.add_row(time_str, f"[{style}]{label}[/]", ev["symbol"],
                     f"{ev['score']:.0f}", dir_dot)

    if not events:
        t_ev.add_row("[dim]No events in last 2h[/]", *[""] * 4)

    console.print(t_ev)
    console.print()

    # ── System Stats ──────────────────────────────────────────────────────
    console.print(
        f"[dim]Events 24h: {stats.get('events_24h', 0):,}  |  "
        f"Snapshots: {stats.get('snapshots', 0):,} ({stats.get('labeled', 0):,} labeled)  |  "
        f"Outcomes: {stats.get('outcomes', 0):,}  |  "
        f"Mem: {stats.get('memory', 'N/A')}[/dim]"
    )
    console.print(f"[dim]Refreshes every {REFRESH_S}s  |  Ctrl+C to quit[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    console = Console()
    console.print("[bold cyan]Connecting to VPS...[/]")

    while True:
        try:
            data = fetch_all()
            btc = get_btc()
            render(console, data, btc)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]  — retrying in 10s")
            time.sleep(10)
            continue

        try:
            time.sleep(REFRESH_S)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/]")
            break

    if _ssh_client:
        _ssh_client.close()


if __name__ == "__main__":
    main()
