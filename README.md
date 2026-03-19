# Bybit Perp Scanner v2

A Python-based scanner that monitors Bybit USDT-M perpetual futures and generates alerts when specific market conditions are detected.

## Features

- **Volume Spike Detection**: Monitors 1-hour quote volume against an EMA baseline
- **Open Interest Tracking**: Detects significant OI changes over the last hour
- **Funding Rate Anomalies**: Alerts on unusual funding rates
- **Basis Calculation**: Shows basis percentage (mark - index)/index for context
- **Multiple Alert Channels**: Console output, Telegram notifications, and webhook support
- **Flexible Configuration**: JSON config file with environment variable overrides
- **Robust Error Handling**: Continues running even if individual API calls fail

## Quick Start

### Windows

```cmd
# Clone or download the project
cd bybit-perp-scanner

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the scanner
python src/main.py
```

### macOS/Linux

```bash
# Clone or download the project
cd bybit-perp-scanner

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Run the scanner
python src/main.py
```

## Configuration

The scanner uses `scanner_config.json` for configuration. You can also override settings using environment variables or a `.env` file.

### Main Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `POLL_SECONDS` | 60 | Seconds between scanning cycles |
| `VOL_EMA_ALPHA` | 0.2 | EMA smoothing factor (0-1) |
| `VOL_SPIKE_RATIO` | 3.0 | Volume spike threshold (vol/EMA) |
| `OI_DELTA_1H_THRESHOLD` | 0.10 | OI change threshold (10%) |
| `FUNDING_ABS_THRESHOLD` | 0.0005 | Funding rate threshold (0.05%) |
| `MIN_TURNOVER24H_USDT` | 20000000 | Minimum 24h volume filter |
| `MAX_SYMBOLS` | 300 | Maximum symbols to monitor |

### Alert Trigger Logic

An alert is triggered when:
1. **Volume spike**: `vol_1h / EMA_1h >= VOL_SPIKE_RATIO`
2. **AND** either:
   - **OI spike**: `OI_delta >= OI_DELTA_1H_THRESHOLD`
   - **OR Funding anomaly**: `abs(funding) >= FUNDING_ABS_THRESHOLD`
3. **Optional basis filter**: If enabled, `abs(basis) >= BASIS_ABS_THRESHOLD`

### Notifications

#### Telegram Setup
1. Create a bot with [@BotFather](https://t.me/botfather)
2. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
3. Configure in `scanner_config.json`:
```json
{
  "USE_TELEGRAM": true,
  "TELEGRAM_BOT_TOKEN": "your_bot_token",
  "TELEGRAM_CHAT_ID": "your_chat_id"
}
```

Or use environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export USE_TELEGRAM=true
```

#### Webhook Setup
```json
{
  "USE_WEBHOOK": true,
  "WEBHOOK_URL": "https://your-webhook-url.com/endpoint"
}
```

### Symbol Filtering

```json
{
  "SYMBOLS_INCLUDE": ["BTCUSDT", "ETHUSDT"],  // Only monitor these (empty = all)
  "SYMBOLS_EXCLUDE": ["DOGEUSDT"],            // Exclude these symbols
  "MIN_TURNOVER24H_USDT": 20000000           // Minimum 24h volume
}
```

## Output Format

Console alerts follow this exact format:
```
[2024-01-15T10:30:00Z] BTCUSDT  vol×=4.25  OIΔ1h=15.30%  funding=0.001250  basis=0.0125%
[2024-01-15T10:30:00Z] ETHUSDT  vol×=3.80  OIΔ1h=8.50%   funding=-0.000750 basis=n/a
— 2 alert(s) this cycle —
```

## Project Structure

```
bybit-perp-scanner/
├── README.md
├── .gitignore
├── requirements.txt
├── scanner_config.json        # Configuration file
├── .env.example              # Environment variables template
└── src/
    ├── main.py               # Entry point
    └── scanner/
        ├── __init__.py
        ├── config.py         # Configuration management
        ├── bybit_api.py      # Bybit API client
        ├── alerts.py         # Alert system
        └── engine.py         # Core scanning logic
```

## API Endpoints Used

- **Symbols**: `GET /v5/market/instruments-info?category=linear`
- **Tickers**: `GET /v5/market/tickers?category=linear`
- **Klines**: `GET /v5/market/kline?category=linear&symbol=...&interval=15&limit=4`
- **Open Interest**: `GET /v5/market/open-interest?category=linear&symbol=...&interval=1h&limit=2`
- **Funding**: `GET /v5/market/history-fund-rate?category=linear&symbol=...&limit=1`

## Troubleshooting

### Common Issues

**"No symbols retrieved from API"**
- Check internet connection
- Bybit API might be temporarily unavailable
- Try running again in a few minutes

**"Configuration validation failed"**
- Check `scanner_config.json` syntax
- Ensure numeric values are within valid ranges
- Verify Telegram/webhook settings if enabled

**High CPU usage**
- Reduce `MAX_SYMBOLS` to monitor fewer pairs
- Increase `POLL_SECONDS` for longer intervals
- Check if too many alerts are being generated

**Missing alerts**
- Lower `VOL_SPIKE_RATIO` for more sensitive detection
- Adjust `OI_DELTA_1H_THRESHOLD` and `FUNDING_ABS_THRESHOLD`
- Temporarily set `VOL_SPIKE_RATIO` to 1.0 to see all activity

### Testing Configuration

To test with more frequent alerts, temporarily modify `scanner_config.json`:
```json
{
  "VOL_SPIKE_RATIO": 1.0,
  "OI_DELTA_1H_THRESHOLD": 0.01,
  "FUNDING_ABS_THRESHOLD": 0.0001
}
```

## Security Notes

- Never commit `.env` files to version control
- Store sensitive tokens in `.env` or environment variables
- Use webhook HTTPS endpoints only
- Regularly rotate API tokens

## Preset Configurations

### Noisy (More Alerts)
```json
{
  "VOL_SPIKE_RATIO": 2.0,
  "OI_DELTA_1H_THRESHOLD": 0.05,
  "FUNDING_ABS_THRESHOLD": 0.0003
}
```

### Quiet (Fewer Alerts)
```json
{
  "VOL_SPIKE_RATIO": 5.0,
  "OI_DELTA_1H_THRESHOLD": 0.20,
  "FUNDING_ABS_THRESHOLD": 0.001
}
```

## Requirements

- Python 3.10+
- Internet connection
- ~50MB RAM usage
- Minimal CPU usage

## License

This project is for educational and personal use. Please respect Bybit's API rate limits and terms of service.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your configuration matches the examples
3. Test with a simple configuration first
4. Check console output for specific error messages
