# Storage Architecture & Data Layout

## Overview

This document defines the optimal storage architecture for the Market Edge Finder system, balancing performance, compression, and production requirements for ~54MB of financial time series data.

## Storage Design Principles

1. **Columnar Compression**: Parquet with Snappy for 2-6x compression
2. **Metadata Tracking**: SQLite for run tracking and orchestration
3. **Temporal Partitioning**: Year/month partitions for efficient walk-forward validation
4. **Atomic Operations**: Safe writes with integrity verification
5. **Production Ready**: Checksums, monitoring, and recovery mechanisms

## Data Size Specifications

### Dataset Calculations (3 years, 20 instruments)
- **Total bars**: 374,400 (18,720 per instrument × 20 instruments)
- **Raw OHLCV**: ~24MB binary, ~72MB CSV
- **Features**: ~15MB (100 features × 18,720 timesteps × float64)
- **Latents**: ~15MB (100-dim latent space × timesteps)
- **Total raw**: ~54MB
- **Compressed (Parquet/Snappy)**: ~12-35MB

## Directory Structure

```
data/
├── raw/                          # Raw OANDA data
│   ├── ohlcv/
│   │   ├── EUR_USD/
│   │   │   ├── year=2022/
│   │   │   │   ├── month=01/
│   │   │   │   │   └── eur_usd_202201.parquet
│   │   │   │   └── month=02/
│   │   │   │       └── eur_usd_202202.parquet
│   │   │   └── year=2023/
│   │   └── GBP_USD/
│   │       └── year=2022/
│   └── metadata/
│       └── data_registry.db      # SQLite metadata
├── processed/                    # Engineered features
│   ├── features/
│   │   ├── year=2022/
│   │   │   ├── month=01/
│   │   │   │   └── features_202201.parquet
│   │   │   └── month=02/
│   │   │       └── features_202202.parquet
│   │   └── year=2023/
│   ├── normalized/
│   │   └── [same structure as features]
│   └── latents/
│       └── [same structure as features]
├── splits/                       # Train/validation splits
│   ├── train/
│   │   ├── EUR_USD.parquet
│   │   └── GBP_USD.parquet
│   └── validation/
│       ├── EUR_USD.parquet
│       └── GBP_USD.parquet
└── cache/                        # Temporary processing files
    └── temp_*.parquet
```

## SQLite Metadata Schema

```sql
-- Data registry for tracking files and integrity
CREATE TABLE data_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    instrument TEXT NOT NULL,
    data_type TEXT NOT NULL,  -- 'ohlcv', 'features', 'latents'
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    checksum TEXT NOT NULL,     -- SHA256
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP
);

-- OANDA API fetch tracking
CREATE TABLE fetch_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    granularity TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    records_fetched INTEGER NOT NULL,
    api_calls_used INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    status TEXT NOT NULL,       -- 'success', 'partial', 'failed'
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing runs and model versions
CREATE TABLE processing_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,     -- 'preprocessing', 'training', 'evaluation'
    config_hash TEXT NOT NULL,  -- Config file hash
    input_files TEXT NOT NULL,  -- JSON array of input file IDs
    output_files TEXT NOT NULL, -- JSON array of output file IDs
    status TEXT NOT NULL,
    duration_seconds REAL,
    log_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model artifacts and versions
CREATE TABLE model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,   -- 'tcnae', 'gbdt', 'context_manager'
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    training_run_id INTEGER,
    metrics TEXT,               -- JSON performance metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (training_run_id) REFERENCES processing_runs(id)
);

-- Create indices for performance
CREATE INDEX idx_data_files_instrument ON data_files(instrument);
CREATE INDEX idx_data_files_type ON data_files(data_type);
CREATE INDEX idx_data_files_date ON data_files(start_date, end_date);
CREATE INDEX idx_fetch_runs_instrument ON fetch_runs(instrument);
CREATE INDEX idx_processing_runs_type ON processing_runs(run_type);
```

## Parquet Schema Definitions

### OHLCV Data Schema
```python
import pyarrow as pa

ohlcv_schema = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    ('open', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('close', pa.float64()),
    ('volume', pa.int64()),
    ('instrument', pa.string()),
    ('year', pa.int32()),
    ('month', pa.int32())
])
```

### Features Schema
```python
# 100 features per timestamp (4×20 local + 20 context)
feature_columns = []
for i in range(20):  # 20 instruments
    for indicator in ['slope_high', 'slope_low', 'volatility', 'direction']:
        feature_columns.append((f'{indicator}_{i}', pa.float64()))

features_schema = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    *feature_columns,
    ('year', pa.int32()),
    ('month', pa.int32())
])
```

### Latents Schema
```python
latent_columns = [(f'latent_{i}', pa.float64()) for i in range(100)]

latents_schema = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    *latent_columns,
    ('year', pa.int32()),
    ('month', pa.int32())
])
```

## Implementation Code

### Storage Manager Class
```python
import sqlite3
import pandas as pd
import pyarrow.parquet as pq
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class StorageManager:
    """
    Production storage manager for Market Edge Finder data.
    
    Handles Parquet files, SQLite metadata, and integrity verification.
    """
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.db_path = self.data_root / "raw" / "metadata" / "data_registry.db"
        
        # Ensure directories exist
        self._create_directory_structure()
        self._initialize_database()
    
    def save_ohlcv_data(self, 
                       data: pd.DataFrame, 
                       instrument: str,
                       compression: str = 'snappy') -> str:
        """
        Save OHLCV data with partitioning and metadata tracking.
        
        Args:
            data: OHLCV DataFrame with timezone-aware timestamps
            instrument: Instrument name (e.g., 'EUR_USD')
            compression: Compression algorithm
            
        Returns:
            Saved file path
        """
        # Add partitioning columns
        data = data.copy()
        data['instrument'] = instrument
        data['year'] = data['timestamp'].dt.year
        data['month'] = data['timestamp'].dt.month
        
        # Determine file path
        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()
        
        file_path = (
            self.data_root / "raw" / "ohlcv" / instrument /
            f"year={start_date.year}" / f"month={start_date.month:02d}" /
            f"{instrument.lower()}_{start_date.strftime('%Y%m')}.parquet"
        )
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically
        temp_path = file_path.with_suffix('.parquet.tmp')
        data.to_parquet(
            temp_path,
            compression=compression,
            index=False,
            partition_cols=None  # Manual partitioning
        )
        
        # Atomic move
        temp_path.rename(file_path)
        
        # Calculate checksum and register
        checksum = self._calculate_checksum(file_path)
        self._register_file(
            file_path=str(file_path.relative_to(self.data_root)),
            instrument=instrument,
            data_type='ohlcv',
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            record_count=len(data),
            file_size=file_path.stat().st_size,
            checksum=checksum
        )
        
        return str(file_path)
    
    def save_features(self, 
                     features: pd.DataFrame,
                     data_type: str = 'features',
                     compression: str = 'snappy') -> str:
        """
        Save feature data with temporal partitioning.
        
        Args:
            features: Features DataFrame with timestamp
            data_type: Type of features ('features', 'normalized', 'latents')
            compression: Compression algorithm
            
        Returns:
            Saved file path
        """
        # Add partitioning columns
        features = features.copy()
        features['year'] = features['timestamp'].dt.year
        features['month'] = features['timestamp'].dt.month
        
        # Group by partition and save
        saved_paths = []
        
        for (year, month), group in features.groupby(['year', 'month']):
            file_path = (
                self.data_root / "processed" / data_type /
                f"year={year}" / f"month={month:02d}" /
                f"{data_type}_{year}{month:02d}.parquet"
            )
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write atomically
            temp_path = file_path.with_suffix('.parquet.tmp')
            group.to_parquet(temp_path, compression=compression, index=False)
            temp_path.rename(file_path)
            
            # Register file
            checksum = self._calculate_checksum(file_path)
            self._register_file(
                file_path=str(file_path.relative_to(self.data_root)),
                instrument='ALL',
                data_type=data_type,
                start_date=group['timestamp'].min().isoformat(),
                end_date=group['timestamp'].max().isoformat(),
                record_count=len(group),
                file_size=file_path.stat().st_size,
                checksum=checksum
            )
            
            saved_paths.append(str(file_path))
        
        return saved_paths
    
    def load_data_by_date_range(self,
                               data_type: str,
                               start_date: datetime,
                               end_date: datetime,
                               instrument: Optional[str] = None) -> pd.DataFrame:
        """
        Load data by date range with efficient partition pruning.
        
        Args:
            data_type: Type of data to load
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            instrument: Optional instrument filter
            
        Returns:
            Combined DataFrame
        """
        # Query metadata for relevant files
        query = """
        SELECT file_path FROM data_files 
        WHERE data_type = ? 
        AND start_date <= ? 
        AND end_date >= ?
        """
        params = [data_type, end_date.isoformat(), start_date.isoformat()]
        
        if instrument:
            query += " AND instrument = ?"
            params.append(instrument)
        
        query += " ORDER BY start_date"
        
        with sqlite3.connect(self.db_path) as conn:
            file_paths = [
                self.data_root / row[0] 
                for row in conn.execute(query, params).fetchall()
            ]
        
        # Load and combine files
        dataframes = []
        for file_path in file_paths:
            df = pd.read_parquet(file_path)
            # Filter to exact date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            if mask.any():
                dataframes.append(df[mask])
        
        if dataframes:
            return pd.concat(dataframes, ignore_index=True).sort_values('timestamp')
        else:
            return pd.DataFrame()
    
    def verify_data_integrity(self) -> Dict[str, bool]:
        """
        Verify integrity of all stored files.
        
        Returns:
            Dictionary of file_path -> integrity_status
        """
        with sqlite3.connect(self.db_path) as conn:
            files = conn.execute(
                "SELECT file_path, checksum FROM data_files"
            ).fetchall()
        
        results = {}
        for file_path, expected_checksum in files:
            full_path = self.data_root / file_path
            if full_path.exists():
                actual_checksum = self._calculate_checksum(full_path)
                results[file_path] = (actual_checksum == expected_checksum)
            else:
                results[file_path] = False
        
        return results
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _register_file(self, **kwargs) -> None:
        """Register file in metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_files 
                (file_path, instrument, data_type, start_date, end_date, 
                 record_count, file_size, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                kwargs['file_path'], kwargs['instrument'], kwargs['data_type'],
                kwargs['start_date'], kwargs['end_date'], kwargs['record_count'],
                kwargs['file_size'], kwargs['checksum']
            ])
    
    def _create_directory_structure(self) -> None:
        """Create the complete directory structure."""
        directories = [
            "raw/ohlcv", "raw/metadata",
            "processed/features", "processed/normalized", "processed/latents",
            "splits/train", "splits/validation",
            "cache"
        ]
        
        for directory in directories:
            (self.data_root / directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Execute schema creation from above
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS data_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    instrument TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_verified TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS fetch_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    records_fetched INTEGER NOT NULL,
                    api_calls_used INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_data_files_instrument ON data_files(instrument);
                CREATE INDEX IF NOT EXISTS idx_data_files_type ON data_files(data_type);
                CREATE INDEX IF NOT EXISTS idx_data_files_date ON data_files(start_date, end_date);
                CREATE INDEX IF NOT EXISTS idx_fetch_runs_instrument ON fetch_runs(instrument);
            """)
```

## Usage Examples

### Basic Storage Operations
```python
from pathlib import Path
from storage import StorageManager

# Initialize storage
storage = StorageManager(Path("./data"))

# Save OHLCV data
ohlcv_data = pd.DataFrame({...})  # From OANDA
file_path = storage.save_ohlcv_data(ohlcv_data, 'EUR_USD')

# Save features
features_data = pd.DataFrame({...})  # Engineered features
storage.save_features(features_data, 'features')

# Load data for training (walk-forward window)
train_data = storage.load_data_by_date_range(
    data_type='features',
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31)
)

# Verify integrity
integrity_results = storage.verify_data_integrity()
```

### Integration with OANDA Downloader
```python
from oanda_downloader import OANDADownloader
from storage import StorageManager

# Download and store
downloader = OANDADownloader(api_key="...", account_id="...")
storage = StorageManager(Path("./data"))

for instrument in instruments:
    data = downloader.fetch_historical(instrument, start, end)
    storage.save_ohlcv_data(data, instrument)
    
    # Track the fetch operation
    storage.log_fetch_run(
        instrument=instrument,
        records_fetched=len(data),
        api_calls_used=downloader.api_calls_made,
        status='success'
    )
```

## Benefits of This Architecture

1. **Compression**: 2-6x reduction in storage space
2. **Performance**: Columnar access patterns optimized for ML workloads
3. **Integrity**: Checksums and atomic writes prevent corruption
4. **Partitioning**: Efficient date-range queries for walk-forward validation
5. **Metadata**: Full audit trail and processing history
6. **Recovery**: Complete tracking enables data lineage and debugging
7. **Production Ready**: Designed for 24/7 operation with monitoring

## Migration from Existing Storage

If migrating from CSV or other formats:

```python
def migrate_csv_to_parquet(csv_dir: Path, storage: StorageManager):
    """Migrate existing CSV files to optimized Parquet storage."""
    
    for csv_file in csv_dir.glob("*.csv"):
        # Parse instrument from filename
        instrument = csv_file.stem.upper()
        
        # Load and convert
        data = pd.read_csv(csv_file, parse_dates=['timestamp'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        
        # Save in new format
        storage.save_ohlcv_data(data, instrument)
        print(f"Migrated {instrument}: {len(data)} records")
```

This storage architecture provides the foundation for a production-ready financial ML system with proper data governance, integrity verification, and optimized access patterns.