#!/usr/bin/env python3
"""
Comprehensive analysis of all 5 indicators across the full dataset (20 FX pairs)
Analyzes: slope_high, slope_low, volatility (ATR/50pip), direction (ADX/100), log_returns
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from configs.instruments import FX_INSTRUMENTS, get_pip_value

def load_all_fx_data():
    """Load all 20 FX pairs from raw data directory"""
    raw_data_dir = project_root / "data/raw"
    all_data = {}
    
    print("📊 Loading FX data from all pairs...")
    
    for instrument in FX_INSTRUMENTS:
        file_path = raw_data_dir / f"{instrument}_3years_H1.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                all_data[instrument] = df
                print(f"✅ {instrument}: {len(df):,} bars loaded")
            except Exception as e:
                print(f"❌ {instrument}: Failed to load - {e}")
        else:
            print(f"⚠️  {instrument}: File not found at {file_path}")
    
    print(f"\n📊 Successfully loaded {len(all_data)} FX pairs")
    return all_data

def analyze_single_instrument(df, instrument, generator):
    """Analyze 5 indicators for a single instrument"""
    print(f"\n🔍 Analyzing {instrument}...")
    
    try:
        # Generate all 5 features
        result_df = generator.generate_features_single_instrument(df, instrument)
        
        # Extract the 5 indicators
        indicators = {
            'slope_high': result_df['slope_high'].values,
            'slope_low': result_df['slope_low'].values,
            'volatility': result_df['volatility'].values,
            'direction': result_df['direction'].values,
            'log_returns': result_df['log_returns'].values
        }
        
        # Calculate statistics for each indicator
        stats = {}
        for name, values in indicators.items():
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                stats[name] = {
                    'count': len(valid_values),
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'p25': np.percentile(valid_values, 25),
                    'p50': np.percentile(valid_values, 50),
                    'p75': np.percentile(valid_values, 75),
                    'coverage': len(valid_values) / len(values) * 100
                }
            else:
                stats[name] = {k: np.nan for k in ['count', 'mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'coverage']}
        
        return stats, indicators, result_df
        
    except Exception as e:
        print(f"❌ Failed to analyze {instrument}: {e}")
        return None, None, None

def create_comprehensive_analysis(all_stats, all_indicators):
    """Create comprehensive statistical analysis across all pairs"""
    
    # Aggregate statistics across all instruments
    aggregated_stats = defaultdict(list)
    
    for instrument, stats in all_stats.items():
        if stats is not None:
            for indicator, indicator_stats in stats.items():
                for stat_name, value in indicator_stats.items():
                    if not np.isnan(value):
                        aggregated_stats[f"{indicator}_{stat_name}"].append(value)
    
    # Create summary table
    summary_df = pd.DataFrame()
    indicators = ['slope_high', 'slope_low', 'volatility', 'direction', 'log_returns']
    
    for indicator in indicators:
        means = [all_stats[inst][indicator]['mean'] for inst in all_stats if all_stats[inst] is not None]
        stds = [all_stats[inst][indicator]['std'] for inst in all_stats if all_stats[inst] is not None]
        coverages = [all_stats[inst][indicator]['coverage'] for inst in all_stats if all_stats[inst] is not None]
        
        means_clean = [x for x in means if not np.isnan(x)]
        stds_clean = [x for x in stds if not np.isnan(x)]
        coverages_clean = [x for x in coverages if not np.isnan(x)]
        
        if means_clean:
            summary_df[indicator] = [
                f"{np.mean(means_clean):.4f}",
                f"{np.std(means_clean):.4f}",
                f"{np.min(means_clean):.4f}",
                f"{np.max(means_clean):.4f}",
                f"{np.mean(stds_clean):.4f}" if stds_clean else "N/A",
                f"{np.mean(coverages_clean):.1f}%" if coverages_clean else "N/A"
            ]
    
    summary_df.index = ['Mean across pairs', 'Std of means', 'Min mean', 'Max mean', 'Avg volatility', 'Avg coverage']
    
    return summary_df

def create_visualization(all_stats, all_indicators):
    """Create comprehensive visualization of all indicators"""
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
    
    indicators = ['slope_high', 'slope_low', 'volatility', 'direction', 'log_returns']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Row 1: Distribution plots for each indicator
    for i, (indicator, color) in enumerate(zip(indicators, colors)):
        ax = fig.add_subplot(gs[0, i if i < 4 else 0])
        
        # Collect all values across instruments
        all_values = []
        for instrument, values in all_indicators.items():
            if values is not None and indicator in values:
                valid_vals = values[indicator][~np.isnan(values[indicator])]
                all_values.extend(valid_vals)
        
        if all_values:
            ax.hist(all_values, bins=50, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'{indicator.replace("_", " ").title()} Distribution\\n({len(all_values):,} values)', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.4f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.6, label=f'+1σ: {mean_val + std_val:.4f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.6, label=f'-1σ: {mean_val - std_val:.4f}')
            ax.legend(fontsize=8)
    
    # Handle 5th indicator in row 2
    if len(indicators) > 4:
        ax = fig.add_subplot(gs[1, 0])
        indicator = indicators[4]
        color = colors[4]
        
        all_values = []
        for instrument, values in all_indicators.items():
            if values is not None and indicator in values:
                valid_vals = values[indicator][~np.isnan(values[indicator])]
                all_values.extend(valid_vals)
        
        if all_values:
            ax.hist(all_values, bins=50, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'{indicator.replace("_", " ").title()} Distribution\\n({len(all_values):,} values)', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.4f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.6, label=f'+1σ: {mean_val + std_val:.4f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.6, label=f'-1σ: {mean_val - std_val:.4f}')
            ax.legend(fontsize=8)
    
    # Row 2-3: Box plots comparing indicators across instruments
    ax_box = fig.add_subplot(gs[1, 1:])
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    
    for indicator in indicators:
        for instrument in sorted(all_indicators.keys()):
            if all_indicators[instrument] is not None and indicator in all_indicators[instrument]:
                valid_vals = all_indicators[instrument][indicator][~np.isnan(all_indicators[instrument][indicator])]
                if len(valid_vals) > 10:  # Only include if sufficient data
                    box_data.append(valid_vals)
                    box_labels.append(f"{instrument}\\n{indicator}")
    
    if box_data:
        bp = ax_box.boxplot(box_data, patch_artist=True, labels=box_labels)
        
        # Color boxes by indicator
        for i, (patch, label) in enumerate(zip(bp['boxes'], box_labels)):
            indicator = label.split('\\n')[1]
            color_idx = indicators.index(indicator)
            patch.set_facecolor(colors[color_idx])
            patch.set_alpha(0.7)
        
        ax_box.set_title('Indicator Values by Instrument and Type', fontweight='bold', fontsize=14)
        ax_box.set_ylabel('Value')
        ax_box.tick_params(axis='x', rotation=45, labelsize=8)
        ax_box.grid(True, alpha=0.3)
    
    # Row 4: Correlation analysis between indicators
    ax_corr = fig.add_subplot(gs[2, :2])
    
    # Calculate correlations across all instruments
    corr_data = {}
    for indicator in indicators:
        corr_data[indicator] = []
        for instrument, values in all_indicators.items():
            if values is not None and indicator in values:
                valid_vals = values[indicator][~np.isnan(values[indicator])]
                # Take sample if too many values
                if len(valid_vals) > 1000:
                    valid_vals = np.random.choice(valid_vals, 1000, replace=False)
                corr_data[indicator].extend(valid_vals)
    
    # Create correlation matrix
    corr_df = pd.DataFrame({k: pd.Series(v) for k, v in corr_data.items()})
    corr_matrix = corr_df.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax_corr, cbar_kws={'shrink': 0.8})
    ax_corr.set_title('Cross-Indicator Correlation Matrix', fontweight='bold', fontsize=14)
    
    # Row 4-5: Volatility analysis (special focus on ATR/50pip)
    ax_vol = fig.add_subplot(gs[2, 2:])
    
    # Analyze volatility by instrument
    vol_by_instrument = {}
    for instrument, values in all_indicators.items():
        if values is not None and 'volatility' in values:
            valid_vals = values['volatility'][~np.isnan(values['volatility'])]
            if len(valid_vals) > 0:
                vol_by_instrument[instrument] = {
                    'mean': np.mean(valid_vals),
                    'std': np.std(valid_vals),
                    'pip_size': get_pip_value(instrument)
                }
    
    # Sort by mean volatility
    sorted_instruments = sorted(vol_by_instrument.keys(), 
                               key=lambda x: vol_by_instrument[x]['mean'], reverse=True)
    
    means = [vol_by_instrument[inst]['mean'] for inst in sorted_instruments]
    stds = [vol_by_instrument[inst]['std'] for inst in sorted_instruments]
    
    x_pos = np.arange(len(sorted_instruments))
    bars = ax_vol.bar(x_pos, means, yerr=stds, capsize=5, 
                      color='red', alpha=0.7, edgecolor='black')
    
    # Add pip interpretation
    for i, (inst, mean_val) in enumerate(zip(sorted_instruments, means)):
        pip_equivalent = mean_val * 50  # Since we use ATR/(pipsize*50)
        ax_vol.text(i, mean_val + stds[i] + 0.01, f'{pip_equivalent:.0f}p', 
                   ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax_vol.set_title('Volatility by Instrument (ATR/50pip)\\nNumbers show pip equivalent', fontweight='bold')
    ax_vol.set_xlabel('FX Pair')
    ax_vol.set_ylabel('ATR/(pipsize*50)')
    ax_vol.set_xticks(x_pos)
    ax_vol.set_xticklabels(sorted_instruments, rotation=45, ha='right')
    ax_vol.grid(True, alpha=0.3)
    
    # Row 5-6: Time series examples
    # Show 3 representative instruments with all 5 indicators
    example_instruments = ['EUR_USD', 'GBP_JPY', 'AUD_NZD']
    
    for idx, instrument in enumerate(example_instruments):
        if instrument in all_indicators and all_indicators[instrument] is not None:
            ax_ts = fig.add_subplot(gs[3 + idx, :])
            
            # Get recent data (last 200 bars)
            values = all_indicators[instrument]
            recent_length = min(200, len(next(iter(values.values()))))
            
            for i, (indicator, color) in enumerate(zip(indicators, colors)):
                if indicator in values:
                    data = values[indicator][-recent_length:]
                    valid_mask = ~np.isnan(data)
                    
                    if np.any(valid_mask):
                        # Normalize for display (0-1 scale)
                        valid_data = data[valid_mask]
                        normalized_data = (valid_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data) + 1e-8)
                        x_indices = np.where(valid_mask)[0]
                        
                        ax_ts.plot(x_indices, normalized_data + i * 1.2, 
                                  color=color, label=f'{indicator} (norm)', linewidth=1.5, alpha=0.8)
            
            ax_ts.set_title(f'{instrument} - All 5 Indicators (Normalized, Last 200 bars)', fontweight='bold')
            ax_ts.set_xlabel('Time Index')
            ax_ts.set_ylabel('Normalized Value + Offset')
            ax_ts.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax_ts.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Analysis: 5 Indicators Across 20 FX Pairs\\n' + 
                 'slope_high, slope_low, volatility (ATR/50pip), direction (ADX/100), log_returns', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def main():
    print("🚀 Starting comprehensive 5-indicator analysis across full dataset...")
    
    # Load all FX data
    all_data = load_all_fx_data()
    
    if not all_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Initialize feature generator
    generator = FXFeatureGenerator()
    
    # Analyze each instrument
    all_stats = {}
    all_indicators = {}
    
    print(f"\n🔍 Analyzing 5 indicators for {len(all_data)} instruments...")
    
    for instrument, df in all_data.items():
        stats, indicators, result_df = analyze_single_instrument(df, instrument, generator)
        all_stats[instrument] = stats
        all_indicators[instrument] = indicators
        
        if stats is not None:
            print(f"✅ {instrument}: Analysis complete")
            # Print key statistics
            for indicator in ['volatility', 'direction', 'log_returns']:
                if indicator in stats:
                    mean_val = stats[indicator]['mean']
                    if indicator == 'volatility':
                        pip_equiv = mean_val * 50
                        print(f"    {indicator}: {mean_val:.3f} (≈{pip_equiv:.1f} pips)")
                    else:
                        print(f"    {indicator}: {mean_val:.4f}")
    
    # Create comprehensive analysis
    print(f"\n📊 Creating comprehensive statistical analysis...")
    summary_df = create_comprehensive_analysis(all_stats, all_indicators)
    
    print(f"\n📈 Summary Statistics Across All Instruments:")
    print("=" * 80)
    print(summary_df.to_string())
    
    # Create visualization
    print(f"\n📊 Creating comprehensive visualization...")
    fig = create_visualization(all_stats, all_indicators)
    
    # Save results
    save_path = project_root / "data/test/comprehensive_5indicators_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive analysis saved to: {save_path}")
    
    # Save summary statistics
    summary_path = project_root / "data/test/indicators_summary_statistics.csv"
    summary_df.to_csv(summary_path)
    print(f"📊 Summary statistics saved to: {summary_path}")
    
    # Additional detailed analysis
    print(f"\n💡 Key Insights:")
    print("=" * 50)
    
    # Volatility insights
    vol_means = [all_stats[inst]['volatility']['mean'] for inst in all_stats 
                 if all_stats[inst] is not None and not np.isnan(all_stats[inst]['volatility']['mean'])]
    if vol_means:
        avg_vol = np.mean(vol_means)
        avg_pips = avg_vol * 50
        print(f"🎯 Average volatility across all pairs: {avg_vol:.3f} (≈{avg_pips:.1f} pips)")
        print(f"   Range: {np.min(vol_means):.3f} to {np.max(vol_means):.3f}")
        print(f"   Std: {np.std(vol_means):.3f}")
    
    # Direction insights  
    dir_means = [all_stats[inst]['direction']['mean'] for inst in all_stats 
                 if all_stats[inst] is not None and not np.isnan(all_stats[inst]['direction']['mean'])]
    if dir_means:
        avg_dir = np.mean(dir_means)
        print(f"🎯 Average direction (ADX/100): {avg_dir:.3f}")
        print(f"   Range: {np.min(dir_means):.3f} to {np.max(dir_means):.3f}")
    
    # Log returns insights
    lr_means = [all_stats[inst]['log_returns']['mean'] for inst in all_stats 
                if all_stats[inst] is not None and not np.isnan(all_stats[inst]['log_returns']['mean'])]
    if lr_means:
        avg_lr = np.mean(lr_means)
        print(f"🎯 Average log returns: {avg_lr:.6f}")
        print(f"   Range: {np.min(lr_means):.6f} to {np.max(lr_means):.6f}")
    
    print(f"\n✅ All 5 indicators successfully analyzed across {len(all_data)} FX pairs!")
    print(f"📊 Total bars analyzed: {sum(len(df) for df in all_data.values()):,}")
    
    plt.show()
    print("🎉 Comprehensive 5-indicator analysis completed!")

if __name__ == "__main__":
    main()