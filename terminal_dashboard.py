#!/usr/bin/env python3
"""
Squeeze Scanner — Live Terminal Dashboard

Reads data from VPS SQLite (single SSH call) + Bybit API for BTC price.
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
    """Reusable SSH connection (reconnects if dead)."""
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
    """Execute command on VPS, return stdout."""
    client = _get_ssh()
    _, stdout, _ = client.exec_command(cmd, timeout=20)
    return stdout.read().decode().strip()


# ── Data fetching (ONE SSH call) ─────────────────────────────────────────────

_MEGA_QUERY = r"""
echo '===FRAGILE===';
sqlite3 -separator '|' DB "
SELECT e.symbol,
       COUNT(DISTINCT e.event_type) as types,
       COUNT(*) as total,
       MAX(e.score) as max_score,
       GROUP_CONCAT(DISTINCT e.event_type) as etypes
FROM events e
WHERE e.timestamp > datetime('now','-2 hours') AND e.score >= 50
GROUP BY e.symbol
ORDER BY types DESC, max_score DESC
LIMIT 12";

echo '===SNAP===';
sqlite3 -separator '|' DB "
SELECT symbol, funding_rate, oi_change_1h_pct, thin_pct,
       bb_width_pct, vacuum_dist_ask, vacuum_dist_bid,
       oi_usd, depth_ask_usdt
FROM ml_snapshots
WHERE timestamp = (SELECT MAX(timestamp) FROM ml_snapshots)
ORDER BY symbol";

echo '===EVENTS===';
sqlite3 -separator '|' DB "
SELECT timestamp, event_type, symbol, score, direction
FROM events
WHERE timestamp > datetime('now','-2 hours')
ORDER BY timestamp DESC
LIMIT 8";

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


def fetch_all() -> Dict[str, Any]:
    """Single SSH call to get all dashboard data."""
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

    # Parse snapshots into dict by symbol
    snap_map: Dict[str, Dict[str, float]] = {}
    for line in sections.get("SNAP", "").split("\n"):
        if not line.strip():
            continue
        p = line.split("|")
        if len(p) >= 9:
            snap_map[p[0]] = {
                "funding_rate": _f(p[1]),
                "oi_change_1h": _f(p[2]),
                "thin_pct": _f(p[3]),
                "bb_width": _f(p[4]),
                "vac_ask": _f(p[5]),
                "vac_bid": _f(p[6]),
                "oi_usd": _f(p[7]),
                "depth_ask": _f(p[8]),
            }

    # Parse fragile coins
    coins: List[Dict[str, Any]] = []
    for line in sections.get("FRAGILE", "").split("\n"):
        if not line.strip():
            continue
        p = line.split("|")
        if len(p) < 5:
            continue
        sym = p[0]
        sd = snap_map.get(sym, {})
        vac = "ok"
        if sd.get("vac_ask", 0) > 5000 or sd.get("vac_bid", 0) > 5000:
            vac = "EMPTY"
        elif sd.get("thin_pct", 0) > 0.90:
            vac = "THIN"

        coins.append({
            "symbol": sym,
            "types": int(_f(p[1])),
            "total": int(_f(p[2])),
            "max_score": _f(p[3]),
            "event_types": p[4],
            "funding_rate": sd.get("funding_rate", 0),
            "oi_change_1h": sd.get("oi_change_1h", 0),
            "thin_pct": sd.get("thin_pct", 0),
            "bb_width": sd.get("bb_width", 0),
            "vacuum": vac,
            "oi_usd": sd.get("oi_usd", 0),
            "depth_ask": sd.get("depth_ask", 0),
        })

    # Parse events
    events: List[Dict[str, str]] = []
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

    # Parse stats
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

    return {
        "coins": coins,
        "snap_map": snap_map,
        "events": events,
        "stats": stats,
    }


def _f(s: str) -> float:
    """Safe float parse."""
    try:
        return float(s.strip()) if s.strip() else 0.0
    except (ValueError, TypeError):
        return 0.0


# ── Bybit API ────────────────────────────────────────────────────────────────


def get_btc() -> Dict[str, float]:
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=5,
        )
        d = r.json()["result"]["list"][0]
        return {
            "price": float(d["lastPrice"]),
            "change_24h": float(d["price24hPcnt"]) * 100,
        }
    except Exception:
        return {"price": 0, "change_24h": 0}


# ── Settlement ────────────────────────────────────────────────────────────────


def next_settlement() -> Tuple[int, int]:
    """Return (settlement_hour, minutes_away)."""
    now = datetime.now(timezone.utc)
    cur = now.hour * 60 + now.minute
    for h in (0, 8, 16, 24):
        m = h * 60
        if m > cur:
            return h % 24, m - cur
    return 0, 1440 - cur


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
    coins = data["coins"]
    events = data["events"]
    stats = data["stats"]
    snap_map = data["snap_map"]

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
    console.print(hdr)
    console.print()

    # ── Section 1: Top Fragile Coins ──────────────────────────────────────
    t1 = Table(
        title="TOP FRAGILE COINS (event-ranked, last 2h)",
        box=box.SIMPLE_HEAVY,
        title_style="bold yellow",
        show_lines=False,
        pad_edge=False,
    )
    t1.add_column("Symbol", style="bold", width=14)
    t1.add_column("Types", justify="center", width=6)
    t1.add_column("Score", justify="right", width=6)
    t1.add_column("Events", width=20)
    t1.add_column("FR%", justify="right", width=8)
    t1.add_column("OI\u03941h", justify="right", width=8)
    t1.add_column("Thin", justify="right", width=6)
    t1.add_column("BB%", justify="right", width=6)
    t1.add_column("Vac", justify="center", width=6)

    _short = {
        "FUNDING_SQUEEZE_SETUP": "FS",
        "VACUUM_BREAK": "VB",
        "VOLUME_EXPLOSION": "VE",
        "CASCADE_ACTIVE": "CA",
        "OI_SURGE": "OI",
        "COMPRESSION_SQUEEZE_SETUP": "CS",
    }

    for c in coins[:8]:
        sym_s = "bold red" if c["types"] >= 4 else "bold yellow" if c["types"] >= 3 else "white"
        fr = c["funding_rate"]
        fr_s = "red" if fr < -0.001 else "green" if fr > 0.001 else "white"
        thin_s = "red" if c["thin_pct"] > 0.90 else "yellow" if c["thin_pct"] > 0.80 else "white"
        oi_s = "yellow" if abs(c["oi_change_1h"]) > 30 else "white"
        vac_s = "red" if c["vacuum"] == "EMPTY" else "yellow" if c["vacuum"] == "THIN" else "dim"

        et_short = ",".join(
            _short.get(e.strip(), e.strip()[:2])
            for e in c["event_types"].split(",")
        )

        t1.add_row(
            f"[{sym_s}]{c['symbol']}[/]",
            f"{c['types']}/5",
            f"{c['max_score']:.0f}",
            et_short,
            f"[{fr_s}]{fr*100:+.3f}%[/]",
            f"[{oi_s}]{c['oi_change_1h']:+.0f}%[/]",
            f"[{thin_s}]{c['thin_pct']:.2f}[/]",
            f"{c['bb_width']:.1f}%",
            f"[{vac_s}]{c['vacuum']}[/]",
        )

    if not coins:
        t1.add_row("[dim]No fragile coins in last 2h[/]", *[""] * 8)

    console.print(t1)
    console.print()

    # ── Section 2: Settlement Ranking ─────────────────────────────────────
    settle_title = f"SETTLEMENT RANKING ({s_hour:02d}:00 in {s_mins}min)"
    if s_mins <= 60:
        settle_title += "  \U0001f525 PRE-SETTLEMENT"

    t2 = Table(
        title=settle_title,
        box=box.SIMPLE_HEAVY,
        title_style="bold magenta",
    )
    t2.add_column("#", width=3)
    t2.add_column("Symbol", width=14)
    t2.add_column("Dir", width=5)
    t2.add_column("FR%", justify="right", width=8)
    t2.add_column("OI $M", justify="right", width=8)
    t2.add_column("Squeeze", width=20)

    # Rank ALL coins from snap_map by squeeze ratio
    ranked: List[Tuple[float, str, Dict]] = []
    for sym, sd in snap_map.items():
        fr = sd.get("funding_rate", 0)
        if fr == 0:
            continue
        sq = abs(fr) * sd.get("oi_usd", 0) / (sd.get("depth_ask", 0) + 1)
        ranked.append((sq, sym, sd))
    ranked.sort(reverse=True)

    for i, (sq, sym, sd) in enumerate(ranked[:5], 1):
        norm = min(100, sq / 10)
        bar_n = int(norm / 10)
        bar = "\u2588" * bar_n + "\u2591" * (10 - bar_n)
        fr = sd["funding_rate"]
        d = "\U0001f7e2 UP" if fr < 0 else "\U0001f534 DN"
        fr_s = "red" if abs(fr) > 0.001 else "white"
        t2.add_row(
            str(i),
            f"[bold]{sym}[/]",
            d,
            f"[{fr_s}]{fr*100:+.3f}%[/]",
            f"${sd.get('oi_usd', 0)/1e6:.1f}M",
            f"{bar} {norm:.0f}",
        )

    if not ranked:
        t2.add_row("[dim]No data[/]", *[""] * 5)

    console.print(t2)
    console.print()

    # ── Section 3: Recent Events ──────────────────────────────────────────
    t3 = Table(
        title="RECENT EVENTS (last 2h)",
        box=box.SIMPLE_HEAVY,
        title_style="bold green",
    )
    t3.add_column("Time", width=6)
    t3.add_column("Type", width=18)
    t3.add_column("Symbol", width=14)
    t3.add_column("Score", justify="right", width=6)
    t3.add_column("Dir", width=4)

    for ev in events[:8]:
        ts = ev["timestamp"]
        try:
            time_str = ts.split("T")[1][:5] if "T" in ts else ts[11:16]
        except Exception:
            time_str = ts[:5]

        et = ev["event_type"]
        label, style = _EVENT_LABEL.get(et, (et[:12], "white"))
        dir_dot = "\U0001f7e2" if ev["direction"] in ("LONG", "UP") else "\U0001f534"

        t3.add_row(
            time_str,
            f"[{style}]{label}[/]",
            ev["symbol"],
            f"{ev['score']:.0f}",
            dir_dot,
        )

    if not events:
        t3.add_row("[dim]No events in last 2h[/]", *[""] * 4)

    console.print(t3)
    console.print()

    # ── Section 4: System Stats ───────────────────────────────────────────
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
