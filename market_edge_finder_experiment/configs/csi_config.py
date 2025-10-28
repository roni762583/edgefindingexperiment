#!/usr/bin/env python3
"""
CSI Configuration Settings

Controls whether to use static configuration or dynamic OANDA API fetching
for CSI parameter calculations.
"""

import os
from typing import Dict, Any

class CSIConfig:
    """CSI calculation configuration"""
    
    def __init__(self):
        # Default to static configuration for reliability
        self.use_dynamic_api = os.getenv("CSI_USE_DYNAMIC_API", "false").lower() == "true"
        self.api_timeout = int(os.getenv("CSI_API_TIMEOUT", "30"))
        self.cache_duration = int(os.getenv("CSI_CACHE_DURATION", "300"))  # 5 minutes
        self.fallback_to_static = True  # Always fall back to static on API failure
        
    def get_settings(self) -> Dict[str, Any]:
        """Get current CSI configuration settings"""
        return {
            "use_dynamic_api": self.use_dynamic_api,
            "api_timeout": self.api_timeout,
            "cache_duration": self.cache_duration,
            "fallback_to_static": self.fallback_to_static
        }
    
    def enable_dynamic_api(self):
        """Enable dynamic OANDA API fetching"""
        self.use_dynamic_api = True
        
    def disable_dynamic_api(self):
        """Disable dynamic OANDA API fetching (use static config)"""
        self.use_dynamic_api = False
        
    def __repr__(self):
        return f"CSIConfig(dynamic={self.use_dynamic_api}, timeout={self.api_timeout}s)"

# Global configuration instance
csi_config = CSIConfig()

def get_csi_config() -> CSIConfig:
    """Get global CSI configuration"""
    return csi_config

def set_dynamic_csi(enabled: bool):
    """Enable/disable dynamic CSI parameter fetching globally"""
    csi_config.use_dynamic_api = enabled

def is_dynamic_csi_enabled() -> bool:
    """Check if dynamic CSI fetching is enabled"""
    return csi_config.use_dynamic_api