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
        hard_filters={'pT': (5, 100), 'id': lambda x: np.abs(x) == 21},
        columns=['id', 'px', 'py', 'pz', 'pT', 'rap'],
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
        hard_id: int,
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
        hard_id : int
            Unique identifier for this hard process
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
        # Pick only positive status particles
        saved_particles = []
        for particle in event_record.particles:
            if particle.status > 0:
                saved_particles.append(particle)
        
        # Build particle dataframe
        particle_records = [
            self._particle_to_dict(particle, soft_event_seed, i)
            for i, particle in enumerate(saved_particles)
        ]
        df = pd.DataFrame(particle_records)

        # Add soft event properties as regular columns
        # Parquet compression handles repetition efficiently
        for key, value in soft_event_props.items():
            df[f'soft_{key}'] = float(value) if isinstance(value, (int, float)) else value  # Makes floats of ints

        # Add the weight
        df['weight'] = event_record.weight

        
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
        
        # Attach metadata to table
        table = table.replace_schema_metadata({
            **table.schema.metadata,
            **{k.encode(): v.encode() for k, v in file_metadata.items()}
        })
        
        # Save to file
        output_file = self.dataset_dir / f"job_{job_id}_{hard_id}_soft_{soft_event_seed}.parquet"
        pq.write_table(table, str(output_file), compression='snappy')
        
        # Cache metadata for fast filtering
        self.metadata_index[str(output_file)] = soft_event_props
        
        logging.info(f"Saved to {output_file.name}: "
                    f"{len(saved_particles)} hard particles, "
                    f"soft v2={soft_event_props.get('v_2', 0):.3f}, "
                    f"soft mult={soft_event_props.get('mult', 0):.1f}")
        
        return str(output_file)

    def filter_and_load(
            self,
            soft_filters: Optional[Dict[str, Any]] = None,
            hard_filters: Optional[Dict[str, Any]] = None,
            columns: Optional[List[str]] = None,
            soft_metadata_subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Multi-level filtering for efficient data loading.

        Parameters:
        -----------
        soft_filters : dict, optional
            Filter soft event properties
        hard_filters : dict, optional
            Filter hard particle properties
        columns : list, optional
            Which particle columns to load
        soft_metadata_subset : list, optional
            If provided, only load these specific soft properties
            Example: ['v_2', 'mult', 'psi_2']

        Returns:
        --------
        pd.DataFrame
        """

        # STAGE 1: Filter files by soft event properties (metadata-only)
        matching_files = self._filter_files_by_soft_props(soft_filters or {})

        logging.info(f"Soft filters matched {len(matching_files)} of "
                     f"{len(list(self.dataset_dir.glob('*.parquet')))} files")

        if not matching_files:
            logging.warning("No files matched soft filters")
            return pd.DataFrame()

        # STAGE 2: Filter files by parquet stats of hard particle properties to see if we can skip whole files
        matching_files = self._apply_hard_filters_with_stats(matching_files, hard_filters or {})

        # STAGE 3: Determine which columns to load
        load_columns = None
        if columns is not None or soft_metadata_subset is not None:
            load_columns = list(columns) if columns else []

            # Always include soft_event_seed for tracking
            if 'soft_event_seed' not in load_columns:
                load_columns.append('soft_event_seed')

            # Add specific soft metadata if requested
            if soft_metadata_subset:
                for prop in soft_metadata_subset:
                    col_name = f'soft_{prop}'
                    if col_name not in load_columns:
                        load_columns.append(col_name)

        # Load particles from matching files
        dfs = []
        for file_path in matching_files:
            try:
                df_chunk = pd.read_parquet(file_path, columns=load_columns)
                dfs.append(df_chunk)
            except Exception as e:
                logging.warning(f"Failed to read {file_path}: {e}")

        if not dfs:
            logging.warning("No particle data loaded")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Report memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        logging.info(f"Loaded {len(df)} particles from {len(matching_files)} files "
                     f"({memory_mb:.1f} MB)")

        # STAGE 4: Apply hard particle filters (row-level)
        if hard_filters:
            df = self._apply_hard_filters(df, hard_filters)
            logging.info(f"After hard filters: {len(df)} particles remaining")

        return df

    def get_soft_metadata_table(
            self,
            soft_filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Get a lightweight table of soft event metadata only.

        Useful for joining with particle data later.
        Much more memory-efficient than loading soft_* columns for all particles.

        Returns:
        --------
        pd.DataFrame with columns: soft_event_seed, v_2, mult etc.
        """

        matching_files = self._filter_files_by_soft_props(soft_filters or {})

        soft_records = []
        for file_path in matching_files:
            table = pq.read_table(str(file_path))
            metadata = table.schema.metadata or {}

            record = {}
            for key, value in metadata.items():
                if isinstance(key, bytes):
                    key = key.decode()
                if isinstance(value, bytes):
                    value = value.decode()

                if key == 'soft_event_seed':
                    record['soft_event_seed'] = int(value)
                elif key.startswith('soft_'):
                    prop_name = key[5:]
                    try:
                        record[prop_name] = float(value)
                    except ValueError:
                        record[prop_name] = value

            soft_records.append(record)

        return pd.DataFrame(soft_records)
    
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

    def _apply_hard_filters_with_stats(self, matching_files: list, hard_filters: dict) -> list:
        """Filter files based on Parquet column statistics first."""
        if not hard_filters:
            return matching_files

        filtered_files = []
        for file_path in matching_files:
            try:
                parquet_file = pq.ParquetFile(str(file_path))
                stats = parquet_file.statistics

                skip_file = False
                for column, criterion in hard_filters.items():
                    if column not in stats:
                        continue

                    col_stats = stats[column]
                    if isinstance(criterion, (tuple, list)) and len(criterion) == 2:
                        # If range is [min_want, max_want] and file has [min_file, max_file]
                        # Skip if: max_file < min_want OR min_file > max_want
                        if col_stats['max'] < criterion[0] or col_stats['min'] > criterion[1]:
                            skip_file = True
                            break

                if not skip_file:
                    filtered_files.append(file_path)

            except Exception as e:
                logging.debug(f"Could not read stats for {file_path}: {e}")
                filtered_files.append(file_path)  # Include on error

        return filtered_files
    
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
