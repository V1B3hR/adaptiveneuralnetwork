"""
Time Series Tracking System for Node State Variables

Provides comprehensive tracking, querying, and visualization capabilities
for key node state variables over time.
"""

import json
import logging
import sqlite3
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from core.time_manager import get_timestamp

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesDataPoint:
    """Single data point in a time series"""
    timestamp: float
    node_id: int
    variable_name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class TimeSeriesQuery:
    """Query specification for time series data"""
    node_ids: Optional[List[int]] = None  # None means all nodes
    variables: Optional[List[str]] = None  # None means all variables
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    max_points: Optional[int] = None
    aggregation: str = "none"  # "none", "mean", "max", "min", "median"
    aggregation_window: float = 1.0  # Time window for aggregation
    
    def matches_point(self, point: TimeSeriesDataPoint) -> bool:
        """Check if a data point matches this query"""
        if self.node_ids is not None and point.node_id not in self.node_ids:
            return False
        if self.variables is not None and point.variable_name not in self.variables:
            return False
        if self.start_time is not None and point.timestamp < self.start_time:
            return False
        if self.end_time is not None and point.timestamp > self.end_time:
            return False
        return True


class TimeSeriesTracker:
    """Comprehensive time series tracking system for node state variables"""
    
    # Key variables to track by default
    DEFAULT_VARIABLES = [
        "energy", "anxiety", "calm", "trust", "phase", 
        "emotional_valence", "arousal", "communication_count"
    ]
    
    def __init__(self, 
                 max_memory_points: int = 10000,
                 persist_to_disk: bool = True,
                 db_path: str = "node_timeseries.db"):
        
        self.max_memory_points = max_memory_points
        self.persist_to_disk = persist_to_disk
        self.db_path = db_path
        
        # In-memory storage for fast access
        self._memory_store: Dict[int, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self._lock = threading.Lock()
        
        # Statistics tracking
        self._stats = {
            "total_points": 0,
            "unique_nodes": set(),
            "unique_variables": set(),
            "first_timestamp": None,
            "last_timestamp": None
        }
        
        # Initialize database if persistent storage is enabled
        if self.persist_to_disk:
            self._init_database()
            
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS time_series (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        node_id INTEGER NOT NULL,
                        variable_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indices for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON time_series(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_node_variable 
                    ON time_series(node_id, variable_name)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_node_time 
                    ON time_series(node_id, timestamp)
                """)
                
                conn.commit()
                logger.info(f"Initialized time series database: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.persist_to_disk = False
            
    def record_node_state(self, node_id: int, state_data: Dict[str, Any], timestamp: Optional[float] = None):
        """
        Record the current state of a node.
        
        Args:
            node_id: Unique identifier for the node
            state_data: Dictionary of variable_name -> value
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = get_timestamp()
            
        with self._lock:
            for variable_name, value in state_data.items():
                if isinstance(value, (int, float)):
                    point = TimeSeriesDataPoint(
                        timestamp=timestamp,
                        node_id=node_id,
                        variable_name=variable_name,
                        value=float(value)
                    )
                    
                    # Add to memory store
                    self._memory_store[node_id][variable_name].append(point)
                    
                    # Update statistics
                    self._stats["total_points"] += 1
                    self._stats["unique_nodes"].add(node_id)
                    self._stats["unique_variables"].add(variable_name)
                    
                    if self._stats["first_timestamp"] is None or timestamp < self._stats["first_timestamp"]:
                        self._stats["first_timestamp"] = timestamp
                    if self._stats["last_timestamp"] is None or timestamp > self._stats["last_timestamp"]:
                        self._stats["last_timestamp"] = timestamp
                    
                    # Persist to database if enabled
                    if self.persist_to_disk:
                        self._persist_point(point)
                        
    def _persist_point(self, point: TimeSeriesDataPoint):
        """Persist a single data point to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                metadata_json = json.dumps(point.metadata) if point.metadata else None
                
                cursor.execute("""
                    INSERT INTO time_series 
                    (timestamp, node_id, variable_name, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (point.timestamp, point.node_id, point.variable_name, 
                     point.value, metadata_json))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to persist data point: {e}")
            
    def query(self, query: TimeSeriesQuery) -> List[TimeSeriesDataPoint]:
        """
        Query time series data based on specified criteria.
        
        Args:
            query: TimeSeriesQuery object specifying what data to retrieve
            
        Returns:
            List of TimeSeriesDataPoint objects matching the query
        """
        results = []
        result_keys = set()  # Track unique results to avoid duplicates
        
        with self._lock:
            # Query from memory store first (fastest)
            for node_id, variables in self._memory_store.items():
                if query.node_ids is None or node_id in query.node_ids:
                    for variable_name, points in variables.items():
                        if query.variables is None or variable_name in query.variables:
                            for point in points:
                                if query.matches_point(point):
                                    # Create unique key for deduplication
                                    key = (point.timestamp, point.node_id, point.variable_name)
                                    if key not in result_keys:
                                        results.append(point)
                                        result_keys.add(key)
            
            # If not enough results and we have persistent storage, query database
            if (len(results) < (query.max_points or 1000) and 
                self.persist_to_disk and 
                (query.start_time is not None or query.end_time is not None)):
                
                db_results = self._query_database(query)
                for point in db_results:
                    key = (point.timestamp, point.node_id, point.variable_name)
                    if key not in result_keys:
                        results.append(point)
                        result_keys.add(key)
                
        # Sort by timestamp
        results.sort(key=lambda p: p.timestamp)
        
        # Apply max_points limit
        if query.max_points and len(results) > query.max_points:
            # Take evenly spaced points
            step = len(results) // query.max_points
            results = results[::max(1, step)][:query.max_points]
            
        return results
        
    def _query_database(self, query: TimeSeriesQuery) -> List[TimeSeriesDataPoint]:
        """Query data from persistent database"""
        results = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build SQL query
                sql_parts = ["SELECT timestamp, node_id, variable_name, value, metadata FROM time_series WHERE 1=1"]
                params = []
                
                if query.node_ids:
                    placeholders = ",".join("?" * len(query.node_ids))
                    sql_parts.append(f"AND node_id IN ({placeholders})")
                    params.extend(query.node_ids)
                    
                if query.variables:
                    placeholders = ",".join("?" * len(query.variables))
                    sql_parts.append(f"AND variable_name IN ({placeholders})")
                    params.extend(query.variables)
                    
                if query.start_time:
                    sql_parts.append("AND timestamp >= ?")
                    params.append(query.start_time)
                    
                if query.end_time:
                    sql_parts.append("AND timestamp <= ?")
                    params.append(query.end_time)
                    
                sql_parts.append("ORDER BY timestamp")
                
                if query.max_points:
                    sql_parts.append("LIMIT ?")
                    params.append(query.max_points)
                    
                sql = " ".join(sql_parts)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    timestamp, node_id, variable_name, value, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else None
                    
                    point = TimeSeriesDataPoint(
                        timestamp=timestamp,
                        node_id=node_id,
                        variable_name=variable_name,
                        value=value,
                        metadata=metadata
                    )
                    results.append(point)
                    
        except Exception as e:
            logger.error(f"Failed to query database: {e}")
            
        return results
        
    def get_latest_values(self, node_id: int, variables: Optional[List[str]] = None) -> Dict[str, float]:
        """Get the latest values for specified variables of a node"""
        latest_values = {}
        
        with self._lock:
            if node_id in self._memory_store:
                variables_to_check = variables or self._memory_store[node_id].keys()
                
                for variable_name in variables_to_check:
                    if variable_name in self._memory_store[node_id]:
                        points = self._memory_store[node_id][variable_name]
                        if points:
                            latest_values[variable_name] = points[-1].value
                            
        return latest_values
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about the tracked data"""
        stats = self._stats.copy()
        stats["unique_nodes"] = list(stats["unique_nodes"])
        stats["unique_variables"] = list(stats["unique_variables"])
        
        if stats["first_timestamp"]:
            stats["first_timestamp_formatted"] = datetime.fromtimestamp(stats["first_timestamp"]).isoformat()
        if stats["last_timestamp"]:
            stats["last_timestamp_formatted"] = datetime.fromtimestamp(stats["last_timestamp"]).isoformat()
            
        # Calculate time range
        if stats["first_timestamp"] and stats["last_timestamp"]:
            stats["time_range_seconds"] = stats["last_timestamp"] - stats["first_timestamp"]
            
        return stats
        
    def visualize_node_variables(self, 
                                node_id: int, 
                                variables: Optional[List[str]] = None,
                                time_range_hours: float = 24,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of node variables over time.
        
        Args:
            node_id: Node to visualize
            variables: Variables to plot (None for default set)
            time_range_hours: How many hours back to show
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if variables is None:
            variables = self.DEFAULT_VARIABLES
            
        end_time = get_timestamp()
        start_time = end_time - (time_range_hours * 3600)
        
        query = TimeSeriesQuery(
            node_ids=[node_id],
            variables=variables,
            start_time=start_time,
            end_time=end_time,
            max_points=1000
        )
        
        data_points = self.query(query)
        
        # Organize data by variable
        variable_data = defaultdict(lambda: {"times": [], "values": []})
        
        for point in data_points:
            dt = datetime.fromtimestamp(point.timestamp)
            variable_data[point.variable_name]["times"].append(dt)
            variable_data[point.variable_name]["values"].append(point.value)
            
        # Create subplots
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2 * n_vars), sharex=True)
        
        if n_vars == 1:
            axes = [axes]
            
        for i, variable in enumerate(variables):
            ax = axes[i]
            data = variable_data[variable]
            
            if data["times"]:
                ax.plot(data["times"], data["values"], marker='o', markersize=2, linewidth=1)
                ax.set_ylabel(variable.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                # Add trend line if enough data
                if len(data["values"]) > 10:
                    timestamps = [t.timestamp() for t in data["times"]]
                    z = np.polyfit(timestamps, data["values"], 1)
                    p = np.poly1d(z)
                    ax.plot(data["times"], p([t.timestamp() for t in data["times"]]), 
                           "r--", alpha=0.7, linewidth=2, label=f'Trend (slope: {z[0]:.4f})')
                    ax.legend()
            else:
                ax.text(0.5, 0.5, f'No data for {variable}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_ylabel(variable.replace('_', ' ').title())
                
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_range_hours/12))))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.suptitle(f'Node {node_id} - State Variables Over Time', y=1.02, fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
            
        return fig
        
    def compare_nodes(self, 
                     node_ids: List[int], 
                     variable: str,
                     time_range_hours: float = 24,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare a single variable across multiple nodes.
        
        Args:
            node_ids: List of nodes to compare
            variable: Variable to compare
            time_range_hours: Time range to show
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        end_time = get_timestamp()
        start_time = end_time - (time_range_hours * 3600)
        
        query = TimeSeriesQuery(
            node_ids=node_ids,
            variables=[variable],
            start_time=start_time,
            end_time=end_time,
            max_points=1000
        )
        
        data_points = self.query(query)
        
        # Organize data by node
        node_data = defaultdict(lambda: {"times": [], "values": []})
        
        for point in data_points:
            dt = datetime.fromtimestamp(point.timestamp)
            node_data[point.node_id]["times"].append(dt)
            node_data[point.node_id]["values"].append(point.value)
            
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for node_id in node_ids:
            data = node_data[node_id]
            if data["times"]:
                ax.plot(data["times"], data["values"], 
                       marker='o', markersize=3, linewidth=2, 
                       label=f'Node {node_id}', alpha=0.8)
                
        ax.set_ylabel(variable.replace('_', ' ').title())
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_range_hours/12))))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.title(f'{variable.replace("_", " ").title()} - Node Comparison', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
            
        return fig
        
    def export_data(self, 
                   query: TimeSeriesQuery, 
                   format: str = "json",
                   output_path: str = "timeseries_export") -> str:
        """
        Export time series data to file.
        
        Args:
            query: Query specifying what data to export
            format: Export format ("json", "csv")
            output_path: Base path for output file
            
        Returns:
            Path to the created file
        """
        data_points = self.query(query)
        
        if format.lower() == "json":
            file_path = f"{output_path}.json"
            export_data = {
                "query": asdict(query),
                "statistics": self.get_statistics(),
                "data_points": [point.to_dict() for point in data_points]
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format.lower() == "csv":
            import csv
            file_path = f"{output_path}.csv"
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "node_id", "variable_name", "value", "metadata"])
                
                for point in data_points:
                    writer.writerow([
                        point.timestamp,
                        point.node_id,
                        point.variable_name,
                        point.value,
                        json.dumps(point.metadata) if point.metadata else ""
                    ])
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Exported {len(data_points)} data points to {file_path}")
        return file_path
        
    def cleanup_old_data(self, keep_hours: float = 168):  # Default: keep 1 week
        """Remove old data to manage storage size"""
        cutoff_time = get_timestamp() - (keep_hours * 3600)
        
        # Clean memory store
        with self._lock:
            for node_id in list(self._memory_store.keys()):
                for variable_name in list(self._memory_store[node_id].keys()):
                    points = self._memory_store[node_id][variable_name]
                    # Filter out old points
                    new_points = deque([p for p in points if p.timestamp >= cutoff_time], 
                                     maxlen=points.maxlen)
                    self._memory_store[node_id][variable_name] = new_points
                    
        # Clean database
        if self.persist_to_disk:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM time_series WHERE timestamp < ?", (cutoff_time,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    logger.info(f"Cleaned up {deleted_count} old database records")
                    
            except Exception as e:
                logger.error(f"Failed to cleanup database: {e}")


# Convenience function for integrating with AliveLoopNode
def track_node_automatically(tracker: TimeSeriesTracker, node, timestamp: Optional[float] = None):
    """
    Automatically extract and track key state variables from an AliveLoopNode.
    
    Args:
        tracker: TimeSeriesTracker instance
        node: AliveLoopNode instance
        timestamp: Optional timestamp
    """
    state_data = {
        "energy": node.energy,
        "anxiety": node.anxiety,
        "phase": hash(node.phase) % 100,  # Convert phase to numeric for tracking
        "communication_count": len(node.signal_history),
        "trust_network_size": len(node.trust_network),
        "memory_count": len(node.memory)
    }
    
    # Add optional attributes if they exist
    if hasattr(node, 'calm'):
        state_data["calm"] = node.calm
    if hasattr(node, 'emotional_state'):
        state_data["emotional_valence"] = node.emotional_state.get("valence", 0.0)
        state_data["arousal"] = node.emotional_state.get("arousal", 0.0)
    if hasattr(node, 'trust_network') and node.trust_network:
        state_data["avg_trust"] = np.mean(list(node.trust_network.values()))
        
    tracker.record_node_state(node.node_id, state_data, timestamp)