"""Bybit API client for perpetual futures data."""

import requests
import time
from typing import Dict, List, Optional, Tuple, Any
import json


class BybitAPIError(Exception):
    """Custom exception for Bybit API errors."""
    pass


class BybitAPI:
    """Client for Bybit v5 public API endpoints."""
    
    BASE_URL = "https://api.bybit.com"
    TIMEOUT = 10
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'bybit-perp-scanner/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a GET request to Bybit API with error handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            BybitAPIError: If API request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Check Bybit API response status
            if data.get('retCode') != 0:
                error_msg = data.get('retMsg', 'Unknown API error')
                raise BybitAPIError(f"API error: {error_msg}")
            
            return data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            raise BybitAPIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise BybitAPIError(f"Invalid JSON response: {str(e)}")
    
    def list_usdt_linear_symbols(self) -> List[str]:
        """
        Get list of active USDT-M perpetual symbols.
        
        Returns:
            Sorted list of symbol names
        """
        try:
            params = {
                'category': 'linear'
            }
            
            data = self._make_request('/v5/market/instruments-info', params)
            symbols = []
            
            for instrument in data.get('list', []):
                if (instrument.get('quoteCoin') == 'USDT' and 
                    instrument.get('status') == 'Trading'):
                    symbols.append(instrument['symbol'])
            
            return sorted(symbols)
            
        except BybitAPIError as e:
            print(f"Error fetching symbols: {e}")
            return []
    
    def get_bulk_tickers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get bulk ticker data for all linear perpetuals.
        
        Returns:
            Dictionary mapping symbol to ticker data with keys:
            - turnover24h: 24h quote volume
            - markPrice: Mark price
            - indexPrice: Index price
        """
        try:
            params = {
                'category': 'linear'
            }
            
            data = self._make_request('/v5/market/tickers', params)
            tickers = {}
            
            for ticker in data.get('list', []):
                symbol = ticker.get('symbol')
                if symbol:
                    tickers[symbol] = {
                        'turnover24h': float(ticker.get('turnover24h', 0)),
                        'markPrice': float(ticker.get('markPrice', 0)) if ticker.get('markPrice') else None,
                        'indexPrice': float(ticker.get('indexPrice', 0)) if ticker.get('indexPrice') else None
                    }
            
            return tickers
            
        except BybitAPIError as e:
            print(f"Error fetching bulk tickers: {e}")
            return {}
    
    def get_1h_quote_volume(self, symbol: str) -> Optional[float]:
        """
        Get 1-hour quote volume by summing last four 15-minute klines.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            1-hour quote volume or None if error
        """
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': '15',
                'limit': 4
            }
            
            data = self._make_request('/v5/market/kline', params)
            klines = data.get('list', [])
            
            if len(klines) < 4:
                return None
            
            # Sum turnover from last 4 klines (index 6 is turnover)
            total_volume = 0.0
            for kline in klines:
                if len(kline) > 6:
                    total_volume += float(kline[6])
            
            return total_volume
            
        except (BybitAPIError, ValueError, IndexError) as e:
            print(f"Error fetching 1h volume for {symbol}: {e}")
            return None
    
    def get_oi_last_prev(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current and previous open interest values.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (current_oi, previous_oi) or (None, None) if error
        """
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'intervalTime': '1h',
                'limit': 2
            }
            
            data = self._make_request('/v5/market/open-interest', params)
            oi_data = data.get('list', [])
            
            if len(oi_data) < 1:
                return None, None
            
            # Most recent OI
            current_oi = float(oi_data[0].get('openInterest', 0))
            
            # Previous OI (if available)
            previous_oi = None
            if len(oi_data) >= 2:
                previous_oi = float(oi_data[1].get('openInterest', 0))
            
            return current_oi, previous_oi
            
        except (BybitAPIError, ValueError, IndexError) as e:
            print(f"Error fetching OI for {symbol}: {e}")
            return None, None
    
    def get_latest_funding(self, symbol: str) -> Optional[float]:
        """
        Get latest funding rate for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Latest funding rate or None if error
        """
        # Use a direct request here so we can handle 404 (Not Found)
        # responses from the funding endpoint gracefully. Some symbols
        # (e.g. newly listed or non-perpetuals) may not have funding
        # history and Bybit returns 404 for those. Treat that as missing
        # funding data rather than an error.
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'limit': 1
            }

            url = f"{self.BASE_URL}/v5/market/funding/history"
            response = self.session.get(url, params=params, timeout=self.TIMEOUT)

            # If the endpoint returns 404 for this symbol, treat as missing data
            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            if data.get('retCode') != 0:
                return None

            funding_data = data.get('result', {}).get('list', [])
            if not funding_data:
                return None

            funding_rate = funding_data[0].get('fundingRate')
            return float(funding_rate) if funding_rate else None

        except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
            print(f"Error fetching funding for {symbol}: {e}")
            return None


# Global API client instance
api_client = BybitAPI()


def list_usdt_linear_symbols() -> List[str]:
    """Get list of active USDT-M perpetual symbols."""
    return api_client.list_usdt_linear_symbols()


def get_bulk_tickers() -> Dict[str, Dict[str, Any]]:
    """Get bulk ticker data for all linear perpetuals."""
    return api_client.get_bulk_tickers()


def get_1h_quote_volume(symbol: str) -> Optional[float]:
    """Get 1-hour quote volume by summing last four 15-minute klines."""
    # Add small delay to avoid rate limiting
    time.sleep(0.01)
    return api_client.get_1h_quote_volume(symbol)


def get_oi_last_prev(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Get current and previous open interest values."""
    # Add small delay to avoid rate limiting
    time.sleep(0.01)
    return api_client.get_oi_last_prev(symbol)


def get_latest_funding(symbol: str) -> Optional[float]:
    """Get latest funding rate for a symbol."""
    # Add small delay to avoid rate limiting
    time.sleep(0.02)
    return api_client.get_latest_funding(symbol)
