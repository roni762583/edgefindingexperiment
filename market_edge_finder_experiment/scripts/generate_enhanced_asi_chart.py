#!/usr/bin/env python3
"""
Generate Enhanced ASI Chart with Trend Lines

Creates a candlestick chart with ASI overlay, swing point markers,
and regression lines for the last 3 HSPs and LSPs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import logging
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Load the processed data with features."""
    logger.info(f"📊 Loading processed data from {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    logger.info(f"✅ Loaded {len(df)} bars with {len(df.columns)} features")
    return df


def get_last_n_swing_points(df: pd.DataFrame, swing_type: str, n: int = 3):
    """Get the last N swing points (HSP or LSP) with their indices and values.
    
    Args:
        df: DataFrame with swing point data
        swing_type: 'hsp' or 'lsp'
        n: Number of last swing points to get
        
    Returns:
        List of tuples (index, asi_value, timestamp)
    """
    if swing_type == 'hsp':
        mask = df['sig_hsp'] == True
    else:
        mask = df['sig_lsp'] == True
    
    swing_indices = df[mask].index
    swing_values = df.loc[swing_indices, 'asi'].values
    
    # Get last n swing points
    if len(swing_indices) >= n:
        last_indices = swing_indices[-n:]
        last_values = swing_values[-n:]
        return [(idx, val, idx) for idx, val in zip(last_indices, last_values)]
    else:
        return [(idx, val, idx) for idx, val in zip(swing_indices, swing_values)]


def draw_trend_line(ax, swing_points, df, color, alpha=0.7, linestyle='-', extend_bars=5):
    """Draw a trend line from the last swing point extending forward only.
    
    Args:
        ax: Matplotlib axis
        swing_points: List of (index, asi_value, timestamp) tuples (should be last 2)
        df: Full dataframe for extending line
        color: Line color
        alpha: Line transparency
        linestyle: Line style
        extend_bars: Number of bars to extend forward
    """
    if len(swing_points) < 2:
        print(f"Not enough swing points: {len(swing_points)}")
        return
    
    # Take only the last 2 swing points to calculate slope
    last_two = swing_points[-2:]
    
    # Convert timestamps to numeric for regression
    timestamps = [mdates.date2num(sp[2]) for sp in last_two]
    asi_values = [sp[1] for sp in last_two]
    
    # Calculate linear regression slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, asi_values)
    
    # Start from the last swing point and extend forward only
    last_timestamp = last_two[-1][2]
    last_asi = last_two[-1][1]
    
    try:
        last_index = df.index.get_loc(last_timestamp)
    except KeyError:
        # Handle case where timestamp not in dataframe
        last_index = len(df) - 1
    
    # Always extend exactly 5 bars ahead from the last swing point
    time_interval = df.index[1] - df.index[0]  # Get time interval between bars
    end_timestamp = last_timestamp + (time_interval * extend_bars)
    
    # Calculate Y values using the regression line equation: y = slope * x + intercept
    first_timestamp = last_two[0][2]
    first_asi = last_two[0][1]
    
    # Calculate Y values at all three points using regression line
    first_x = mdates.date2num(first_timestamp)
    last_x = mdates.date2num(last_timestamp)
    end_x = mdates.date2num(end_timestamp)
    
    first_y = slope * first_x + intercept
    last_y = slope * last_x + intercept
    end_y = slope * end_x + intercept
    
    # Draw the regression line from first swing point through last and extending forward
    ax.plot([first_timestamp, end_timestamp], [first_y, end_y], 
            color=color, alpha=0.9, linestyle=linestyle, linewidth=4.0,
            label=f'{color.capitalize()} Trend Line')
    
    # Mark both swing points with circles
    ax.scatter([first_timestamp, last_timestamp], [first_asi, last_asi], 
               color=color, s=100, marker='o', alpha=0.8, 
               edgecolor='black', linewidth=1)
    
    return slope, intercept, r_value


def create_enhanced_asi_chart(df: pd.DataFrame, output_path: Path, zoom_date: str = "2025-01-22"):
    """Create enhanced candlestick chart with ASI overlay and trend lines.
    
    Args:
        df: DataFrame with OHLC and ASI data
        output_path: Path to save the chart
        zoom_date: Optional date to zoom in on (format: 'YYYY-MM-DD' or 'MM-DD')
    """
    
    # Show from 01/21 onwards to see more context for regression lines
    try:
        zoom_start = pd.to_datetime('2025-01-21', utc=True)
        zoom_mask = (df.index >= zoom_start)
        if zoom_mask.any():
            df_plot = df[zoom_mask].copy()
            title_suffix = f" - From 01/21 onwards"
            logger.info(f"🔍 Filtering to 01/21 onwards: {len(df_plot)} bars")
        else:
            df_plot = df.copy()
            title_suffix = f" - No 01/21 data found, showing all"
            logger.warning(f"⚠️ No 01/21 data found, showing full dataset")
    except Exception as e:
        df_plot = df.copy()
        title_suffix = f" - Filter error, showing all"
        logger.warning(f"⚠️ Filter error: {e}")
        
    # Enable zoom features for better trend line visibility
    zoom_date = True
    
    # Set up the figure and subplots with shared X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1], sharex=True)
    fig.suptitle(f'AUD/CHF Sample Data - Candlestick Chart with ASI{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data from filtered dataframe
    times = df_plot.index
    opens = df_plot['open'].values
    highs = df_plot['high'].values
    lows = df_plot['low'].values
    closes = df_plot['close'].values
    asi_values = df_plot['asi'].values
    
    # Price statistics
    price_range = highs.max() - lows.min()
    price_change = closes[-1] - opens[0]
    price_change_pct = (price_change / opens[0]) * 100
    
    # Plot candlesticks
    logger.info("📈 Plotting candlestick chart...")
    for i in range(len(df_plot)):
        x = times[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        # Color: green if close > open, red otherwise
        color = 'green' if close_price > open_price else 'red'
        
        # High-low line
        ax1.plot([x, x], [low_price, high_price], color='black', linewidth=0.8)
        
        # Candlestick body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        # Calculate proper width for bars (time difference between bars)
        if i < len(df_plot) - 1:
            time_diff = (times[i+1] - times[i]).total_seconds() / (24 * 3600)  # Convert to days
            width = time_diff * 0.6  # 60% of time interval
        else:
            width = 0.025  # Default width for last bar
        
        rect = Rectangle((mdates.date2num(x) - width/2, body_bottom), 
                        width, body_height, 
                        facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
        ax1.add_patch(rect)
    
    # Add close price line
    ax1.plot(times, closes, color='red', linewidth=1.5, alpha=0.8, label='Close Price')
    
    # Price chart formatting
    ax1.set_ylabel('Price (AUD/CHF)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add price statistics box
    stats_text = f"""Price Statistics:
Open: {opens[0]:.5f}
Close: {closes[-1]:.5f}
High: {highs.max():.5f}
Low: {lows.min():.5f}
Change: {price_change_pct:+.2f}%
Range: {price_range:.5f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Plot ASI as line and dots
    logger.info("📊 Plotting ASI with swing points and trend lines...")
    ax2.plot(times, asi_values, color='blue', linewidth=1.2, alpha=0.8, label='ASI Line')
    
    # Plot HSP (High Swing Points) if they exist
    hsp_mask = df_plot['sig_hsp'] == True
    if hsp_mask.any():
        hsp_times = times[hsp_mask]
        hsp_values = df_plot.loc[hsp_mask, 'asi']
        
        # HSP markers (upward triangles)
        ax2.scatter(hsp_times, hsp_values, color='red', s=40, marker='^', 
                   label=f'HSP ({len(hsp_values)})', zorder=5, 
                   edgecolors='darkred', linewidth=0.8)
    
    # Plot LSP (Low Swing Points) if they exist  
    lsp_mask = df_plot['sig_lsp'] == True
    if lsp_mask.any():
        lsp_times = times[lsp_mask]
        lsp_values = df_plot.loc[lsp_mask, 'asi']
        
        # LSP markers (downward triangles)
        ax2.scatter(lsp_times, lsp_values, color='green', s=40, marker='v', 
                   label=f'LSP ({len(lsp_values)})', zorder=5,
                   edgecolors='darkgreen', linewidth=0.8)
    
    # Get and draw adjacent regression lines for last 3 HSPs and LSPs
    logger.info("📈 Drawing adjacent regression lines for HSPs and LSPs...")
    
    # IMPORTANT: Use full dataset (df) to find swing points, not filtered data (df_plot)
    # Get last 3 HSPs to draw 2 adjacent regression lines
    last_3_hsps = get_last_n_swing_points(df, 'hsp', 3)
    if len(last_3_hsps) >= 3:
        # Line 1: HSP1 to HSP2 (older pair)
        hsp_pair_1 = last_3_hsps[:2]  # First two HSPs
        draw_trend_line(ax2, hsp_pair_1, df, 'red', alpha=0.6, linestyle='--')
        
        # Line 2: HSP2 to HSP3 (newer pair) 
        hsp_pair_2 = last_3_hsps[1:]  # Last two HSPs
        draw_trend_line(ax2, hsp_pair_2, df, 'darkred', alpha=0.9, linestyle='--')
    
    # Get last 3 LSPs to draw 2 adjacent regression lines
    last_3_lsps = get_last_n_swing_points(df, 'lsp', 3)
    if len(last_3_lsps) >= 3:
        # Line 1: LSP1 to LSP2 (older pair)
        lsp_pair_1 = last_3_lsps[:2]  # First two LSPs
        draw_trend_line(ax2, lsp_pair_1, df, 'green', alpha=0.6, linestyle='--')
        
        # Line 2: LSP2 to LSP3 (newer pair)
        lsp_pair_2 = last_3_lsps[1:]  # Last two LSPs  
        draw_trend_line(ax2, lsp_pair_2, df, 'darkgreen', alpha=0.9, linestyle='--')
    
    ax2.set_ylabel('ASI Value', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Extend x-axis to show the projected trend lines
    if zoom_date and len(df_plot) < 20:  # Only for zoomed views
        time_interval = df_plot.index[1] - df_plot.index[0]
        extended_end = df_plot.index[-1] + (time_interval * 6)  # Extra buffer
        ax2.set_xlim(df_plot.index[0], extended_end)
        ax1.set_xlim(df_plot.index[0], extended_end)
    
    # ASI statistics
    asi_min, asi_max = asi_values.min(), asi_values.max()
    asi_range = asi_max - asi_min
    
    asi_stats_text = f"""ASI Statistics:
Start: {asi_values[0]:.6f}
Final: {asi_values[-1]:.6f}
Min: {asi_min:.6f}
Max: {asi_max:.6f}
Range: {asi_range:.6f}
Mean: {asi_values.mean():.6f}
Std: {asi_values.std():.6f}"""
    
    ax2.text(0.02, 0.98, asi_stats_text, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Format x-axis - only show labels on bottom subplot due to sharex=True
    # Determine appropriate time formatting based on data range
    time_span = times[-1] - times[0]
    if time_span.days <= 1:
        # For single day, show hours
        date_format = '%H:%M'
        locator = mdates.HourLocator(interval=2)
    elif time_span.days <= 7:
        # For week, show day-hour
        date_format = '%m-%d %H:%M'
        locator = mdates.HourLocator(interval=12)
    else:
        # For longer periods, show just dates
        date_format = '%m-%d'
        locator = mdates.DayLocator(interval=max(1, time_span.days // 10))
    
    # Apply formatting only to bottom subplot (sharex handles alignment)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax2.xaxis.set_major_locator(locator)
    
    # Rotate labels only on bottom subplot
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot titles
    ax1.set_title('Price Chart (Candlesticks)', fontsize=14, fontweight='bold')
    ax2.set_title('Accumulation Swing Index (ASI) with Swing Points', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Chart saved to {output_path}")
    
    # Display chart (disabled to prevent hanging)
    # plt.show()
    
    return fig


def main(zoom_date: str = None):
    """Main function to generate enhanced ASI chart.
    
    Args:
        zoom_date: Optional date to zoom in on (format: 'MM-DD' or 'YYYY-MM-DD')
    """
    
    logger.info("🚀 Starting enhanced ASI chart generation...")
    
    # File paths
    input_file = Path("data/test/processed_sample_data.csv")
    
    # Generate output filename based on zoom
    if zoom_date:
        zoom_suffix = zoom_date.replace('-', '')
        output_chart = Path(f"data/test/enhanced_asi_chart_zoom_{zoom_suffix}.png")
    else:
        output_chart = Path("data/test/enhanced_asi_chart.png")
    
    # Check if input file exists
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        logger.info("💡 Run 'python3 scripts/process_instrument_data.py data/test/sample_data.csv' first")
        return
    
    try:
        # Load processed data
        df = load_processed_data(input_file)
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'asi', 'sig_hsp', 'sig_lsp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return
        
        # Create and save enhanced chart with trend lines
        fig = create_enhanced_asi_chart(df, output_chart, zoom_date)
        
        logger.info("🎉 Enhanced ASI chart generation completed successfully!")
        logger.info(f"📊 Chart saved as: {output_chart}")
        
    except Exception as e:
        logger.error(f"❌ Chart generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    # Check for zoom date argument
    zoom_date = None
    if len(sys.argv) > 1:
        zoom_date = sys.argv[1]
        logger.info(f"📅 Zoom date requested: {zoom_date}")
    
    main(zoom_date)