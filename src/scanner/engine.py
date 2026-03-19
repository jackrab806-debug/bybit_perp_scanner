"""Core scanning engine with EMA tracking and signal detection."""

import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from . import bybit_api
from .alerts import AlertManager


class VolumeEMATracker:
    """Tracks exponential moving average of volume for each symbol."""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.emas: Dict[str, float] = {}
    
    def update(self, symbol: str, volume: float) -> float:
        """
        Update EMA for a symbol and return the current EMA value.
        
        Args:
            symbol: Trading pair symbol
            volume: Current volume value
            
        Returns:
            Current EMA value
        """
        if symbol not in self.emas:
            # Initialize EMA with first volume value
            self.emas[symbol] = volume
        else:
            # Update EMA: EMA = α * current + (1-α) * previous_EMA
            self.emas[symbol] = self.alpha * volume + (1 - self.alpha) * self.emas[symbol]
        
        return self.emas[symbol]
    
    def get_ema(self, symbol: str) -> Optional[float]:
        """Get current EMA for a symbol."""
        return self.emas.get(symbol)


class ScannerEngine:
    """Main scanning engine that orchestrates the monitoring process."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.volume_tracker = VolumeEMATracker(config["VOL_EMA_ALPHA"])
        self.alert_manager = AlertManager(config)
        self.symbols_cache: List[str] = []
        self.last_symbol_refresh = 0
        self.symbol_refresh_interval = 3600  # Refresh symbols every hour
        # Debug flag (can be set via env var DEBUG=true)
        self.debug = bool(self.config.get('DEBUG', False))
    
    def _refresh_symbols(self) -> List[str]:
        """
        Refresh the list of symbols to monitor.
        
        Returns:
            List of symbols to monitor
        """
        current_time = time.time()
        
        # Only refresh if cache is empty or interval has passed
        if (not self.symbols_cache or 
            current_time - self.last_symbol_refresh > self.symbol_refresh_interval):
            
            print("Refreshing symbol list...")
            all_symbols = bybit_api.list_usdt_linear_symbols()
            
            if not all_symbols:
                print("Warning: No symbols retrieved from API")
                return self.symbols_cache  # Return cached symbols if API fails
            
            # Get bulk tickers for filtering
            tickers = bybit_api.get_bulk_tickers()
            
            # Filter symbols by turnover
            filtered_symbols = []
            min_turnover = self.config["MIN_TURNOVER24H_USDT"]
            
            for symbol in all_symbols:
                ticker = tickers.get(symbol)
                if ticker and ticker.get('turnover24h', 0) >= min_turnover:
                    filtered_symbols.append(symbol)
            
            # Apply include/exclude filters
            if self.config["SYMBOLS_INCLUDE"]:
                # If include list is specified, only use those symbols
                filtered_symbols = [s for s in filtered_symbols 
                                  if s in self.config["SYMBOLS_INCLUDE"]]
            
            if self.config["SYMBOLS_EXCLUDE"]:
                # Remove excluded symbols
                filtered_symbols = [s for s in filtered_symbols 
                                  if s not in self.config["SYMBOLS_EXCLUDE"]]
            
            # Limit number of symbols
            max_symbols = self.config["MAX_SYMBOLS"]
            if len(filtered_symbols) > max_symbols:
                filtered_symbols = filtered_symbols[:max_symbols]
            
            self.symbols_cache = filtered_symbols
            self.last_symbol_refresh = current_time
            
            print(f"Symbol list refreshed: {len(self.symbols_cache)} symbols")
        
        return self.symbols_cache
    
    def _calculate_basis(self, mark_price: Optional[float], 
                        index_price: Optional[float]) -> Optional[float]:
        """
        Calculate basis percentage.
        
        Args:
            mark_price: Mark price
            index_price: Index price
            
        Returns:
            Basis percentage or None if calculation not possible
        """
        if mark_price is None or index_price is None or index_price == 0:
            return None
        
        return ((mark_price - index_price) / index_price) * 100
    
    def _check_trigger_conditions(self, symbol: str, vol_ratio: float, 
                                 oi_delta: float, funding: Optional[float], 
                                 basis: Optional[float]) -> bool:
        """
        Check if alert trigger conditions are met.
        
        Args:
            symbol: Trading pair symbol
            vol_ratio: Volume spike ratio
            oi_delta: Open interest delta percentage
            funding: Funding rate
            basis: Basis percentage
            
        Returns:
            True if conditions are met, False otherwise
        """
        # Primary condition: volume spike
        if vol_ratio < self.config["VOL_SPIKE_RATIO"]:
            return False
        
        # Secondary conditions: OI spike OR funding anomaly
        oi_condition = oi_delta >= self.config["OI_DELTA_1H_THRESHOLD"]
        
        funding_condition = False
        if funding is not None:
            funding_condition = abs(funding) >= self.config["FUNDING_ABS_THRESHOLD"]
        
        if not (oi_condition or funding_condition):
            return False
        
        # Optional basis filter
        if self.config["USE_BASIS_FILTER"] and basis is not None:
            if abs(basis) < self.config["BASIS_ABS_THRESHOLD"]:
                return False
        
        return True
    
    def _process_symbol(self, symbol: str, tickers: Dict[str, Dict[str, Any]]) -> None:
        """
        Process a single symbol for alerts.
        
        Args:
            symbol: Trading pair symbol
            tickers: Bulk ticker data
        """
        try:
            # Get ticker data
            ticker = tickers.get(symbol)
            if not ticker:
                return
            
            # Get 1-hour volume
            vol_1h = bybit_api.get_1h_quote_volume(symbol)
            if vol_1h is None or vol_1h <= 0:
                return
            
            # Update and get EMA
            ema_1h = self.volume_tracker.update(symbol, vol_1h)
            
            # Calculate volume ratio
            if ema_1h <= 0:
                return
            vol_ratio = vol_1h / ema_1h
            
            # Get open interest data
            oi_current, oi_previous = bybit_api.get_oi_last_prev(symbol)
            
            # Calculate OI delta
            oi_delta = 0.0
            if oi_current is not None and oi_previous is not None and oi_previous > 0:
                oi_delta = ((oi_current - oi_previous) / oi_previous) * 100
            
            # Get funding rate
            funding = bybit_api.get_latest_funding(symbol)
            
            # Calculate basis
            basis = self._calculate_basis(ticker.get('markPrice'), ticker.get('indexPrice'))
            
            # Check trigger conditions
            # If debug is enabled, print per-symbol metrics to help diagnose
            if self.debug:
                print(f"DEBUG: {symbol} vol_1h={vol_1h:.2f} ema_1h={ema_1h:.2f} vol_ratio={vol_ratio:.2f} oi_delta={oi_delta:.2f}% funding={funding if funding is not None else 'n/a'} basis={basis if basis is not None else 'n/a'}")

            if self._check_trigger_conditions(symbol, vol_ratio, oi_delta, funding, basis):
                self.alert_manager.send_alert(symbol, vol_ratio, oi_delta, funding, basis)
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    def run_cycle(self) -> None:
        """Run one complete scanning cycle."""
        try:
            # Refresh symbols list
            symbols = self._refresh_symbols()
            
            if not symbols:
                print("No symbols to monitor")
                return
            
            # Get bulk tickers once per cycle
            print(f"Fetching data for {len(symbols)} symbols...")
            tickers = bybit_api.get_bulk_tickers()
            
            if not tickers:
                print("Failed to fetch ticker data")
                return
            
            # Process each symbol
            for symbol in symbols:
                self._process_symbol(symbol, tickers)
            
            # End cycle and print summary
            self.alert_manager.end_cycle()
        
        except Exception as e:
            error_msg = f"Cycle error: {e}"
            self.alert_manager.send_error_notification(error_msg)
    
    def run_loop(self) -> None:
        """Run the main scanning loop."""
        try:
            # Initial setup
            symbols = self._refresh_symbols()
            self.alert_manager.send_startup_message(len(symbols))
            
            # Main loop
            while True:
                cycle_start = time.time()
                
                # Run scanning cycle
                self.run_cycle()
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.config["POLL_SECONDS"] - cycle_duration)
                
                if sleep_time > 0:
                    print(f"Cycle completed in {cycle_duration:.1f}s, sleeping for {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Cycle took {cycle_duration:.1f}s (longer than {self.config['POLL_SECONDS']}s interval)")
        
        except KeyboardInterrupt:
            print("\nScanner stopped by user")
        except Exception as e:
            error_msg = f"Fatal error: {e}"
            self.alert_manager.send_error_notification(error_msg)
            raise


def run_loop(config: Dict[str, Any]) -> None:
    """
    Main entry point for running the scanner.
    
    Args:
        config: Configuration dictionary
    """
    engine = ScannerEngine(config)
    engine.run_loop()
