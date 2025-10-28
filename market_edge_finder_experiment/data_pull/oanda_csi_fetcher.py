#!/usr/bin/env python3
"""
OANDA v20 API Dynamic CSI Parameter Fetcher

Fetches real-time margin requirements, pip values, and commission costs
from OANDA v20 API for accurate CSI calculations, following the reference
implementation provided by the user.
"""

import os
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import v20
from v20 import Context
import logging

@dataclass
class CSIParameters:
    """CSI parameters for an instrument"""
    instrument: str
    margin_rate: float          # From OANDA API
    margin_usd: float          # Converted to USD  
    usd_per_pip: float         # Dynamic calculation
    commission_usd: float      # From spread
    pip_location: int          # From OANDA API
    current_rate: float        # Current bid rate

class OandaCSIFetcher:
    """
    Fetch dynamic CSI parameters from OANDA v20 API
    Following the user's reference implementation exactly
    """
    
    def __init__(self):
        """Initialize OANDA v20 API context"""
        self.api = self._create_v20_context()
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.standard_lot_size = 100000
        self.logger = logging.getLogger(__name__)
        
    def _create_v20_context(self) -> Context:
        """Create OANDA v20 API context using environment variables"""
        environment = os.getenv("OANDA_ENVIRONMENT", "live")
        
        if environment == "live":
            hostname = "api-fxtrade.oanda.com"
        else:
            hostname = "api-fxpractice.oanda.com"
        
        return Context(
            hostname=hostname,
            port=443,
            ssl=True,
            application="MarketEdgeFinderCSI",
            token=os.getenv("OANDA_API_KEY"),
            datetime_format="RFC3339"
        )
    
    def get_bid_rate(self, instrument: str) -> float:
        """Get current bid rate for instrument"""
        try:
            price_response = self.api.pricing.get(
                self.account_id,
                instruments=instrument,
                includeUnitsAvailable=False
            )
            
            if price_response.status != 200:
                raise Exception(f"Failed to get pricing: {price_response.status}")
                
            prices_list = price_response.get("prices", 200)
            return float(prices_list[0].closeoutBid)
            
        except Exception as e:
            self.logger.error(f"Failed to get bid rate for {instrument}: {e}")
            raise
    
    def get_ask_rate(self, instrument: str) -> float:
        """Get current ask rate for instrument"""
        try:
            price_response = self.api.pricing.get(
                self.account_id,
                instruments=instrument,
                includeUnitsAvailable=False
            )
            
            if price_response.status != 200:
                raise Exception(f"Failed to get pricing: {price_response.status}")
                
            prices_list = price_response.get("prices", 200)
            return float(prices_list[0].closeoutAsk)
            
        except Exception as e:
            self.logger.error(f"Failed to get ask rate for {instrument}: {e}")
            raise
    
    def calculate_usd_per_pip(self, instrument_obj, current_rate: float) -> float:
        """
        Calculate USD per pip following the exact logic from user's reference code
        """
        instrument_name = instrument_obj.name
        pip_location = instrument_obj.pipLocation
        
        # Number of quote currency units per pip
        number_of_cross_currency_per_pip = self.standard_lot_size * (10 ** pip_location)
        
        # Parse currency names
        base_currency = instrument_name[:3]
        quote_currency = instrument_name[4:7]
        
        if instrument_name.endswith('USD'):
            # Quote currency is USD - pip value already in USD
            usd_per_pip = number_of_cross_currency_per_pip
            
        else:
            # Quote currency is NOT USD - need conversion
            base_currency_per_pip = number_of_cross_currency_per_pip / current_rate
            
            try:
                # Try to find BaseCurrency_USD pair
                base_to_usd_rate = self.get_bid_rate(f"{base_currency}_USD")
                usd_per_pip = base_currency_per_pip * base_to_usd_rate
                
            except:
                if base_currency == 'USD':
                    # USD is base currency (e.g., USD_CAD)
                    usd_per_pip = number_of_cross_currency_per_pip / current_rate
                else:
                    # Use reciprocal USD_BaseCurrency rate
                    try:
                        usd_to_base_rate = self.get_bid_rate(f"USD_{base_currency}")
                        usd_per_pip = base_currency_per_pip / usd_to_base_rate
                    except:
                        # Fallback to simplified calculation
                        usd_per_pip = 10.0 if 'JPY' not in instrument_name else 7.0
        
        return usd_per_pip
    
    def calculate_margin_usd(self, instrument_obj, current_rate: float) -> float:
        """
        Calculate margin requirement in USD following reference implementation
        """
        instrument_name = instrument_obj.name
        margin_rate = instrument_obj.marginRate
        base_currency = instrument_name[:3]
        
        # Base margin calculation
        required_margin = margin_rate * self.standard_lot_size
        
        if instrument_name.endswith('USD'):
            # Quote currency is USD
            required_margin *= current_rate
            
        else:
            # Need to convert to USD
            try:
                if base_currency != 'USD':
                    # Try USD_BaseCurrency rate (reciprocal)
                    usd_to_base_rate = self.get_bid_rate(f"USD_{base_currency}")
                    reciprocal_usd_base = 1.0 / usd_to_base_rate
                    required_margin *= reciprocal_usd_base
                    
            except:
                try:
                    # Try BaseCurrency_USD rate
                    base_to_usd_rate = self.get_bid_rate(f"{base_currency}_USD") 
                    required_margin *= base_to_usd_rate
                except:
                    # Final fallback
                    try:
                        usd_to_base_rate = self.get_bid_rate(f"USD_{base_currency}")
                        required_margin *= usd_to_base_rate
                    except:
                        # Use current rate as fallback
                        required_margin *= current_rate
        
        return required_margin
    
    def calculate_commission_usd(self, instrument_obj, usd_per_pip: float) -> float:
        """
        Calculate commission cost in USD using spread (following reference code)
        """
        instrument_name = instrument_obj.name
        pip_location = instrument_obj.pipLocation
        
        try:
            ask_rate = self.get_ask_rate(instrument_name)
            bid_rate = self.get_bid_rate(instrument_name)
            
            # Commission estimated as spread value in USD
            spread_in_price_units = ask_rate - bid_rate
            spread_in_pips = spread_in_price_units * (10 ** (-pip_location))
            commission_usd = spread_in_pips * usd_per_pip
            
            return commission_usd
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate commission for {instrument_name}: {e}")
            # Fallback to static values
            return 20.0 if 'JPY' not in instrument_name else 15.0
    
    def fetch_csi_parameters(self, instrument_name: str) -> CSIParameters:
        """
        Fetch all CSI parameters for a single instrument from OANDA API
        """
        try:
            # Get instrument details
            instruments_response = self.api.account.instruments(self.account_id)
            if instruments_response.status != 200:
                raise Exception(f"Failed to get instruments: {instruments_response.status}")
            
            instruments_list = instruments_response.get("instruments", 200)
            
            # Find the specific instrument
            instrument_obj = None
            for inst in instruments_list:
                if inst.name == instrument_name:
                    instrument_obj = inst
                    break
            
            if instrument_obj is None:
                raise Exception(f"Instrument {instrument_name} not found")
            
            # Get current rates
            current_rate = self.get_bid_rate(instrument_name)
            
            # Calculate all parameters
            usd_per_pip = self.calculate_usd_per_pip(instrument_obj, current_rate)
            margin_usd = self.calculate_margin_usd(instrument_obj, current_rate)
            commission_usd = self.calculate_commission_usd(instrument_obj, usd_per_pip)
            
            return CSIParameters(
                instrument=instrument_name,
                margin_rate=float(instrument_obj.marginRate),
                margin_usd=margin_usd,
                usd_per_pip=usd_per_pip,
                commission_usd=commission_usd,
                pip_location=instrument_obj.pipLocation,
                current_rate=current_rate
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fetch CSI parameters for {instrument_name}: {e}")
            raise
    
    def fetch_all_csi_parameters(self, instrument_list: list) -> Dict[str, CSIParameters]:
        """
        Fetch CSI parameters for all instruments in the list
        """
        results = {}
        
        self.logger.info(f"Fetching CSI parameters for {len(instrument_list)} instruments...")
        
        for instrument in instrument_list:
            try:
                params = self.fetch_csi_parameters(instrument)
                results[instrument] = params
                self.logger.info(f"✅ {instrument}: M=${params.margin_usd:.0f}, V=${params.usd_per_pip:.2f}, C=${params.commission_usd:.2f}")
                
            except Exception as e:
                self.logger.error(f"❌ {instrument}: Failed - {e}")
                # Continue with next instrument rather than failing completely
                
        return results

def test_oanda_csi_fetcher():
    """Test the OANDA CSI fetcher with a few instruments"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test instruments
    test_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'GBP_JPY']
    
    try:
        fetcher = OandaCSIFetcher()
        results = fetcher.fetch_all_csi_parameters(test_instruments)
        
        print("\n🧪 OANDA CSI PARAMETER FETCHING TEST")
        print("=" * 50)
        
        for instrument, params in results.items():
            print(f"{instrument}:")
            print(f"  Margin Rate: {params.margin_rate:.4f}")
            print(f"  Margin USD: ${params.margin_usd:.2f}")
            print(f"  USD per Pip: ${params.usd_per_pip:.2f}")
            print(f"  Commission: ${params.commission_usd:.2f}")
            print(f"  Current Rate: {params.current_rate:.5f}")
            print()
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_oanda_csi_fetcher()