#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Test with our actual ASI data
df = pd.read_csv('/Users/shmuelzbaida/Desktop/Aharon2025/edgefindingexperiment/market_edge_finder_experiment/data/test/sample_data.csv')
asi = df['AUD_CHF_asi'].values

n = len(asi)
temp_df = pd.DataFrame({'asi': asi})

local_hsp = np.full(n, False)
local_lsp = np.full(n, False)
for i in range(1, n-1):
    if temp_df['asi'][i] > temp_df['asi'][i-1] and temp_df['asi'][i] > temp_df['asi'][i+1]:
        local_hsp[i] = True
    if temp_df['asi'][i] < temp_df['asi'][i-1] and temp_df['asi'][i] < temp_df['asi'][i+1]:
        local_lsp[i] = True

sig_hsp = np.full(n, False)
sig_lsp = np.full(n, False)
last_hsp_idx = None
last_lsp_idx = None
for i in range(1, n-1):
    if local_hsp[i]:
        if last_hsp_idx is None or (last_lsp_idx is not None and last_lsp_idx > last_hsp_idx):
            sig_hsp[i] = True
            last_hsp_idx = i
    if local_lsp[i]:
        if last_lsp_idx is None or (last_hsp_idx is not None and last_hsp_idx > last_lsp_idx):
            sig_lsp[i] = True
            last_lsp_idx = i

# Print results
swing_points = []
for i in range(n):
    if sig_hsp[i]:
        swing_points.append((i, 'H', asi[i]))
    if sig_lsp[i]:
        swing_points.append((i, 'L', asi[i]))

swing_points.sort()

print(f'Total swing points: {len(swing_points)}')
for i, (idx, swing_type, value) in enumerate(swing_points):
    print(f'{i+1}. Index {idx}: {swing_type}SP = {value:.2f}')

sequence = [sp[1] for sp in swing_points]
print(f'Sequence: {"-".join(sequence)}')

# Check alternating
is_alternating = True
for i in range(1, len(sequence)):
    if sequence[i] == sequence[i-1]:
        is_alternating = False
        print(f'ERROR: Non-alternating at positions {i} and {i+1}: {sequence[i-1]}-{sequence[i]}')

print(f'Is properly alternating: {is_alternating}')