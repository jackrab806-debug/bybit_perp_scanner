#!/usr/bin/env python3
"""
Bybit Perp Scanner v2 - Main Entry Point

A Python scanner that monitors Bybit USDT-M perpetuals and alerts on:
- 1h quote volume spikes vs EMA baseline
- Open Interest jumps over the last hour  
- Funding rate anomalies
- Shows basis% for informational purposes

Usage:
    python src/main.py
    python -m src.main
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scanner.config import load_config, validate_config
from src.scanner.engine import run_loop


def main():
    """Main entry point for the Bybit Perp Scanner."""
    print("=" * 50)
    print("Bybit Perp Scanner v2")
    print("=" * 50)
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        
        # Validate configuration
        if not validate_config(config):
            print("Configuration validation failed. Please check your settings.")
            sys.exit(1)
        
        print("Configuration loaded successfully")
        print(f"Poll interval: {config['POLL_SECONDS']}s")
        print(f"Volume spike threshold: {config['VOL_SPIKE_RATIO']}x")
        print(f"OI delta threshold: {config['OI_DELTA_1H_THRESHOLD']*100:.1f}%")
        print(f"Funding threshold: {config['FUNDING_ABS_THRESHOLD']*100:.3f}%")
        print(f"Min 24h turnover: ${config['MIN_TURNOVER24H_USDT']:,}")
        print(f"Max symbols: {config['MAX_SYMBOLS']}")
        
        if config['USE_TELEGRAM']:
            print("✓ Telegram notifications enabled")
        if config['USE_WEBHOOK']:
            print("✓ Webhook notifications enabled")
        if config['USE_BASIS_FILTER']:
            print(f"✓ Basis filter enabled (threshold: {config['BASIS_ABS_THRESHOLD']*100:.3f}%)")
        
        print("-" * 50)
        
        # Start the scanner
        run_loop(config)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
