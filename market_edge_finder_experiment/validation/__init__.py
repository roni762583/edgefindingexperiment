"""
Monte Carlo Validation Framework for Edge Discovery

Implements Dr. Howard Bandy's statistical methodology adapted for FX prediction validation.
Provides rigorous statistical testing to determine if discovered edges are significant.
"""

from .edge_discovery_monte_carlo import EdgeMonteCarloValidator, validate_edge_discovery

__all__ = ['EdgeMonteCarloValidator', 'validate_edge_discovery']