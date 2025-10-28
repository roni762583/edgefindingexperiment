#!/usr/bin/env python3
"""
Simple swing detection implementation similar to reference code
This is an alternative to the complex Wilder method for practical trading analysis
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class SimpleSwingState:
    """State for simple swing detection (much simpler than Wilder)"""
    asi_history: List[float]
    bar_count: int
    
    # Detected swing points (indices and values)
    hsp_indices: List[int] 
    hsp_values: List[float]
    lsp_indices: List[int]
    lsp_values: List[float]
    
    # Optional filtering state
    last_confirmed_hsp_asi: Optional[float] = None
    last_confirmed_lsp_asi: Optional[float] = None
    
    def __post_init__(self):
        if not hasattr(self, 'asi_history'):
            self.asi_history = []
        if not hasattr(self, 'bar_count'):
            self.bar_count = 0
        if not hasattr(self, 'hsp_indices'):
            self.hsp_indices = []
        if not hasattr(self, 'hsp_values'):
            self.hsp_values = []
        if not hasattr(self, 'lsp_indices'):
            self.lsp_indices = []
        if not hasattr(self, 'lsp_values'):
            self.lsp_values = []

class SimpleSwingDetector:
    """
    Simple swing detection based on 3-bar patterns
    Similar to reference code - much more practical than complex Wilder method
    """
    
    def __init__(self, min_distance: int = 1, use_exceeding_filter: bool = True):
        """
        Args:
            min_distance: Minimum bars between same-type swing points
            use_exceeding_filter: Apply exceeding extremes filter for quality
        """
        self.min_distance = min_distance
        self.use_exceeding_filter = use_exceeding_filter
        self.logger = logging.getLogger(__name__)
    
    def detect_swing_batch(self, asi_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch swing detection on complete ASI series
        
        Args:
            asi_values: Complete ASI time series
            
        Returns:
            hsp_flags: Boolean array marking HSP positions
            lsp_flags: Boolean array marking LSP positions
        """
        n = len(asi_values)
        hsp_flags = np.full(n, False)
        lsp_flags = np.full(n, False)
        
        # Step 1: Find all 3-bar patterns
        hsp_candidates = []
        lsp_candidates = []
        
        for i in range(1, n-1):
            if np.isnan(asi_values[i-1]) or np.isnan(asi_values[i]) or np.isnan(asi_values[i+1]):
                continue
                
            # HSP: local maximum
            if asi_values[i] > asi_values[i-1] and asi_values[i] > asi_values[i+1]:
                hsp_candidates.append(i)
            
            # LSP: local minimum  
            if asi_values[i] < asi_values[i-1] and asi_values[i] < asi_values[i+1]:
                lsp_candidates.append(i)
        
        # Step 2: Apply minimum distance filter
        hsp_filtered = self._apply_min_distance_filter(hsp_candidates, asi_values)
        lsp_filtered = self._apply_min_distance_filter(lsp_candidates, asi_values)
        
        # Step 3: Apply exceeding extremes filter if enabled
        if self.use_exceeding_filter:
            hsp_final, lsp_final = self._apply_exceeding_filter(hsp_filtered, lsp_filtered, asi_values)
        else:
            hsp_final, lsp_final = hsp_filtered, lsp_filtered
        
        # Convert to boolean arrays
        for idx in hsp_final:
            hsp_flags[idx] = True
        for idx in lsp_final:
            lsp_flags[idx] = True
            
        return hsp_flags, lsp_flags
    
    def detect_swing_incremental(self, asi_value: float, state: SimpleSwingState) -> Tuple[bool, bool]:
        """
        Incremental swing detection for live processing
        
        Args:
            asi_value: Current ASI value
            state: Swing detection state
            
        Returns:
            detected_hsp: True if HSP detected at current bar
            detected_lsp: True if LSP detected at current bar
        """
        # Add current ASI to history
        state.asi_history.append(asi_value)
        current_idx = state.bar_count
        state.bar_count += 1
        
        detected_hsp = False
        detected_lsp = False
        
        # Need at least 3 bars for pattern detection
        if len(state.asi_history) >= 3:
            # Check 3-bar pattern for middle bar (current_idx - 1)
            left_asi = state.asi_history[-3]
            middle_asi = state.asi_history[-2] 
            right_asi = state.asi_history[-1]  # Current
            middle_idx = current_idx - 1
            
            # Skip NaN values
            if not (np.isnan(left_asi) or np.isnan(middle_asi) or np.isnan(right_asi)):
                
                # HSP: middle bar higher than both neighbors
                if middle_asi > left_asi and middle_asi > right_asi:
                    if self._passes_min_distance(middle_idx, state.hsp_indices, 'HSP'):
                        if not self.use_exceeding_filter or self._passes_exceeding_filter(middle_asi, 'HSP', state):
                            state.hsp_indices.append(middle_idx)
                            state.hsp_values.append(middle_asi)
                            state.last_confirmed_hsp_asi = middle_asi
                            detected_hsp = True
                
                # LSP: middle bar lower than both neighbors  
                if middle_asi < left_asi and middle_asi < right_asi:
                    if self._passes_min_distance(middle_idx, state.lsp_indices, 'LSP'):
                        if not self.use_exceeding_filter or self._passes_exceeding_filter(middle_asi, 'LSP', state):
                            state.lsp_indices.append(middle_idx)
                            state.lsp_values.append(middle_asi)
                            state.last_confirmed_lsp_asi = middle_asi
                            detected_lsp = True
        
        return detected_hsp, detected_lsp
    
    def _apply_min_distance_filter(self, candidates: List[int], asi_values: np.ndarray) -> List[int]:
        """Apply minimum distance constraint between same-type swing points"""
        if not candidates or self.min_distance <= 1:
            return candidates
        
        filtered = []
        
        for idx in candidates:
            # Check distance from last swing of same type
            if not filtered or (idx - filtered[-1]) >= self.min_distance:
                filtered.append(idx)
            else:
                # Keep the stronger swing if within min_distance
                last_idx = filtered[-1]
                if abs(asi_values[idx]) > abs(asi_values[last_idx]):
                    filtered[-1] = idx  # Replace with stronger swing
        
        return filtered
    
    def _apply_exceeding_filter(self, hsp_indices: List[int], lsp_indices: List[int], 
                               asi_values: np.ndarray) -> Tuple[List[int], List[int]]:
        """Apply exceeding extremes filter (reference code style)"""
        # Combine and sort all swings by index
        all_swings = []
        for idx in hsp_indices:
            all_swings.append((idx, 'H', asi_values[idx]))
        for idx in lsp_indices:
            all_swings.append((idx, 'L', asi_values[idx]))
        
        all_swings.sort(key=lambda x: x[0])
        
        if not all_swings:
            return [], []
        
        hsp_filtered = []
        lsp_filtered = []
        
        # Keep first swing point
        first_idx, first_type, first_asi = all_swings[0]
        if first_type == 'H':
            hsp_filtered.append(first_idx)
            last_hsp_asi = first_asi
            last_lsp_asi = None
        else:
            lsp_filtered.append(first_idx)
            last_lsp_asi = first_asi
            last_hsp_asi = None
        
        # Filter subsequent swings
        for idx, swing_type, asi_val in all_swings[1:]:
            
            if swing_type == 'H':
                # Keep HSP if it's higher than last HSP or first HSP after LSP
                if last_hsp_asi is None or asi_val > last_hsp_asi:
                    hsp_filtered.append(idx)
                    last_hsp_asi = asi_val
                    
            else:  # swing_type == 'L'
                # Keep LSP if it's lower than last LSP or first LSP after HSP
                if last_lsp_asi is None or asi_val < last_lsp_asi:
                    lsp_filtered.append(idx)
                    last_lsp_asi = asi_val
        
        return hsp_filtered, lsp_filtered
    
    def _passes_min_distance(self, current_idx: int, previous_indices: List[int], swing_type: str) -> bool:
        """Check if current swing passes minimum distance constraint"""
        if not previous_indices or self.min_distance <= 1:
            return True
        
        last_idx = previous_indices[-1]
        return (current_idx - last_idx) >= self.min_distance
    
    def _passes_exceeding_filter(self, current_asi: float, swing_type: str, state: SimpleSwingState) -> bool:
        """Check if current swing passes exceeding extremes filter"""
        if swing_type == 'HSP':
            # HSP must exceed last confirmed HSP or be first HSP
            return (state.last_confirmed_hsp_asi is None or 
                   current_asi > state.last_confirmed_hsp_asi)
        else:  # LSP
            # LSP must be lower than last confirmed LSP or be first LSP
            return (state.last_confirmed_lsp_asi is None or 
                   current_asi < state.last_confirmed_lsp_asi)

# Integration functions for existing codebase
def detect_simple_swings_batch(asi_values: np.ndarray, min_distance: int = 1, 
                              use_exceeding_filter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for batch simple swing detection
    
    Args:
        asi_values: ASI time series
        min_distance: Minimum bars between same-type swings
        use_exceeding_filter: Apply quality filter
        
    Returns:
        hsp_flags: Boolean array marking HSP positions
        lsp_flags: Boolean array marking LSP positions
    """
    detector = SimpleSwingDetector(min_distance=min_distance, use_exceeding_filter=use_exceeding_filter)
    return detector.detect_swing_batch(asi_values)

def create_simple_swing_state() -> SimpleSwingState:
    """Create initial state for incremental simple swing detection"""
    return SimpleSwingState(
        asi_history=[],
        bar_count=0,
        hsp_indices=[],
        hsp_values=[],
        lsp_indices=[],
        lsp_values=[]
    )