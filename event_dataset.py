"""
Hierarchical event dataset management for efficient multi-level filtering.

Provides tools for saving hard particle events with soft event metadata
and loading subsets based on soft event properties (v2, multiplicity, etc.)
and hard particle properties (pT, rapidity, etc.).

Usage:
    dataset = HierarchicalEventDataset("results/particle_dataset")
    
    # Load with multi-level filtering
    df = dataset.filter_and_load(
        soft_filters={'v_2': (0.1, 0.3), 'urqmd_dNch_deta': (200, 400)},
        hard_filters={'pT': (5, 100), 'pdg_id': lambda x: np.abs(x) == 21},
        columns=['pdg_id', 'px', 'py', 'pz', 'pT', 'rap'],
    )
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import fields, is_dataclass


class HierarchicalEventDataset:
    """
    Efficient multi-level filtering for hard particle events.
    
    Separates soft event properties (v2, multiplicity, impact parameter, etc.)
    from hard particle properties (pT, rapidity, PDG ID, etc.) to enable
    fast filtering without loading unnecessary data.
    
    Structure:
    - Hard particle columns stored in Parquet data
    - Soft event metadata stored in Parquet file-level metadata
    - Two-stage filtering: soft properties first (file-level), then hard properties (row-level)
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize dataset manager.
        
        Parameters:
        -----------
        dataset_dir : str
            Directory where Parquet files will be stored
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_index = {}  # Map: file_path -> soft_event_metadata
    
    def save_job_output(
        self,
        job_id: int,
        soft_event_seed: int,
        event_record: Any,
        soft_event_props: dict,
        config_dict: dict,
    ) -> str:
        """
        Save hard particles to Parquet with soft event metadata.
        
        Parameters:
        -----------
        job_id : int
            Unique identifier for this HTC job
        soft_event_seed : int
            Identifier for the soft sector event
        event_record : EventRecord
            EventRecord object containing particles
        soft_event_props : dict
            Properties of soft event (v_2, multiplicity, impact_parameter, etc.)
        config_dict : dict
            Configuration settings dictionary
        
        Returns:
        --------
        str
            Path to saved Parquet file
        """
        
        # Build particle dataframe
        particle_records = [
            self._particle_to_dict(particle, soft_event_seed, i)
            for i, particle in enumerate(event_record.particles)
        ]
        df = pd.DataFrame(particle_records)
        
        # Optimize dtypes for compression
        self._optimize_dtypes(df)
        
        # Create PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Store soft event properties in file metadata
        file_metadata = {
            'soft_event_seed': str(soft_event_seed),
            'job_id': str(job_id),
        }
        
        # Add all soft event properties to metadata
        for key, value in soft_event_props.items():
            file_metadata[f'soft_{key}'] = str(value)
        
        # Add config hash for reference
        # Can't flatten dict to tuple, so hash doesn't work...
        # file_metadata['config_hash'] = str(hash(tuple(sorted(config_dict.items()))))
        
        # Attach metadata to table
        table = table.replace_schema_metadata({
            **table.schema.metadata,
            **{k.encode(): v.encode() for k, v in file_metadata.items()}
        })
        
        # Save to file
        output_file = self.dataset_dir / f"job_{job_id:06d}_soft_{soft_event_seed:06d}.parquet"
        pq.write_table(table, str(output_file), compression='snappy')
        
        # Cache metadata for fast filtering
        self.metadata_index[str(output_file)] = soft_event_props
        
        logging.info(f"Saved to {output_file.name}: "
                    f"{len(event_record.particles)} particles, "
                    f"v2={soft_event_props.get('v_2', 0):.3f}, "
                    f"mult={soft_event_props.get('urqmd_dNch_deta', 0):.1f}")
        
        return str(output_file)
    
    def filter_and_load(
        self,
        soft_filters: Optional[Dict[str, Any]] = None,
        hard_filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Multi-level filtering for efficient data loading.
        
        STAGE 1: Filter files by soft event properties (metadata-only, very fast)
        STAGE 2: Load particles from matching files (with column selection)
        STAGE 3: Apply hard particle filters (row-level)
        
        Parameters:
        -----------
        soft_filters : dict, optional
            Filter soft event properties. Filters are applied FIRST to reduce
            files read before loading any particle data.
            
            Examples:
                {'v_2': (0.1, 0.3)}
                {'urqmd_dNch_deta': (200, 400)}
                {'impact_parameter': (0, 5)}
            
            Supports:
            - Range filters (tuple/list of length 2): inclusive on both ends
            - Exact match filters (single value)
        
        hard_filters : dict, optional
            Filter hard particle properties. Applied AFTER loading, row-level.
            
            Examples:
                {'pT': (5, 100)}
                {'rap': (-1, 1)}
                {'pdg_id': lambda x: np.abs(x) == 21}  # Custom functions
            
            Supports:
            - Range filters: (min, max) inclusive
            - Callable filters: any function that returns boolean array
        
        columns : list, optional
            Which particle columns to load (None = all).
            Loading fewer columns saves memory and I/O.
            
            Example: ['pdg_id', 'px', 'py', 'pz', 'pT', 'rap']
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all particles matching both filter levels
        
        Examples:
        ---------
        # Soft filter only
        df = dataset.filter_and_load(
            soft_filters={'v_2': (0.1, 0.3)},
        )
        
        # Soft + hard filters with column selection
        df = dataset.filter_and_load(
            soft_filters={
                'urqmd_dNch_deta': (200, 400),
                'impact_parameter': (0, 5),
            },
            hard_filters={
                'pT': (5, 100),
                'pdg_id': lambda x: np.abs(x) == 21,  # Gluons
            },
            columns=['pdg_id', 'px', 'py', 'pz', 'pT', 'rap'],
        )
        
        # Complex particle selection
        def is_charged_pion(pdg_id):
            return np.abs(pdg_id) == 211
        
        df = dataset.filter_and_load(
            soft_filters={'v_2': (0.1, 0.2)},
            hard_filters={
                'pdg_id': is_charged_pion,
                'rap': lambda x: np.abs(x) < 2.4,
            },
        )
        """
        
        # STAGE 1: Filter files by soft event properties (metadata-only)
        matching_files = self._filter_files_by_soft_props(soft_filters or {})
        
        logging.info(f"Soft filters matched {len(matching_files)} of "
                    f"{len(list(self.dataset_dir.glob('*.parquet')))} files")
        
        if not matching_files:
            logging.warning("No files matched soft filters")
            return pd.DataFrame()
        
        # STAGE 2: Load particles from matching files with column selection
        dfs = []
        for file_path in matching_files:
            try:
                df_chunk = pd.read_parquet(file_path, columns=columns)
                dfs.append(df_chunk)
            except Exception as e:
                logging.warning(f"Failed to read {file_path}: {e}")
        
        if not dfs:
            logging.warning("No particle data loaded")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Loaded {len(df)} particles from {len(matching_files)} files")
        
        # STAGE 3: Apply hard particle filters (row-level)
        if hard_filters:
            df = self._apply_hard_filters(df, hard_filters)
            logging.info(f"After hard filters: {len(df)} particles remaining")
        
        return df
    
    def _filter_files_by_soft_props(self, soft_filters: dict) -> List[str]:
        """
        Fast file-level filtering using cached metadata.
        No file I/O needed here!
        """
        matching_files = []
        
        for file_path, soft_props in self.metadata_index.items():
            if self._soft_props_match(soft_props, soft_filters):
                matching_files.append(file_path)
        
        # If metadata index is empty, scan files
        if not self.metadata_index:
            matching_files = self._scan_and_cache_metadata(soft_filters)
        
        return matching_files
    
    def _scan_and_cache_metadata(self, soft_filters: dict) -> List[str]:
        """
        Scan Parquet files and cache metadata on first run.
        Very fast since we only read metadata, not data.
        """
        parquet_files = sorted(self.dataset_dir.glob("*.parquet"))
        matching_files = []
        
        logging.info(f"Scanning metadata from {len(parquet_files)} files...")
        
        for file_path in parquet_files:
            try:
                # Read only schema/metadata (O(1) operation, very fast)
                table = pq.read_table(str(file_path))
                metadata = table.schema.metadata or {}
                
                # Parse soft event properties from metadata
                soft_props = {}
                for key, value in metadata.items():
                    if isinstance(key, bytes):
                        key = key.decode()
                    if isinstance(value, bytes):
                        value = value.decode()
                    
                    if key.startswith('soft_'):
                        prop_name = key[5:]  # Remove 'soft_' prefix
                        try:
                            soft_props[prop_name] = float(value)
                        except ValueError:
                            soft_props[prop_name] = value
                
                self.metadata_index[str(file_path)] = soft_props
                
                # Check if matches filters
                if self._soft_props_match(soft_props, soft_filters):
                    matching_files.append(str(file_path))
            
            except Exception as e:
                logging.warning(f"Failed to read {file_path}: {e}")
        
        logging.info(f"Cached metadata for {len(self.metadata_index)} files")
        return matching_files
    
    @staticmethod
    def _soft_props_match(soft_props: dict, filters: dict) -> bool:
        """Check if soft event properties match all filter criteria."""
        if not filters:
            return True
        
        for key, criterion in filters.items():
            if key not in soft_props:
                return False
            
            value = soft_props[key]
            
            if isinstance(criterion, (tuple, list)) and len(criterion) == 2:
                # Range filter (inclusive)
                if not (criterion[0] <= value <= criterion[1]):
                    return False
            else:
                # Exact match
                if value != criterion:
                    return False
        
        return True
    
    @staticmethod
    def _apply_hard_filters(df: pd.DataFrame, hard_filters: dict) -> pd.DataFrame:
        """
        Apply row-level filters to hard particle data.
        Filters can be ranges or callables.
        """
        filtered_df = df.copy()
        
        for column, criterion in hard_filters.items():
            if column not in filtered_df.columns:
                logging.warning(f"Column '{column}' not found in data")
                continue
            
            if callable(criterion):
                # Custom filter function
                mask = criterion(filtered_df[column])
                filtered_df = filtered_df[mask]
            elif isinstance(criterion, (tuple, list)) and len(criterion) == 2:
                # Range filter (inclusive)
                mask = (filtered_df[column] >= criterion[0]) & \
                       (filtered_df[column] <= criterion[1])
                filtered_df = filtered_df[mask]
            else:
                # Exact match
                filtered_df = filtered_df[filtered_df[column] == criterion]
        
        return filtered_df
    
    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> None:
        """In-place dtype optimization for compression."""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val >= -32768 and max_val <= 32767:
                    df[col] = df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df[col] = df[col].astype('int32')
    
    @staticmethod
    def _particle_to_dict(particle: Any, soft_event_seed: int, particle_idx: int) -> dict:
        """Generic conversion of Particle to flat dictionary."""
        record = {
            'soft_event_seed': soft_event_seed,
            'particle_idx': particle_idx,
        }
        
        # Extract all dataclass fields
        if is_dataclass(particle):
            for field_obj in fields(particle):
                field_name = field_obj.name
                field_value = getattr(particle, field_name)
                
                # Skip complex types
                if isinstance(field_value, (list, dict)):
                    continue
                
                record[field_name] = field_value
        
        # Add computed properties
        for prop_name in ['pT', 'mT', 'E', 'rap']:
            if prop_name not in record:
                try:
                    record[prop_name] = getattr(particle, prop_name)
                except AttributeError:
                    pass
        
        # Add path length
        try:
            record['path_length'] = particle.pathlength_since(particle.tau_0)
        except AttributeError:
            pass
        
        return record
