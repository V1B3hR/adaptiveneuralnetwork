"""
Database integration for SQL and NoSQL databases.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Optional database dependencies
try:
    import sqlalchemy as sa
    from sqlalchemy import (
        JSON,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        Text,
        create_engine,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class Base:
        pass
    def Column(*args, **kwargs):
        return None
    Integer = String = Float = DateTime = Text = JSON = None

try:
    import pymongo
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    # SQL Database config
    sql_url: str | None = None  # e.g., "postgresql://user:pass@localhost/db"
    sql_pool_size: int = 10
    sql_max_overflow: int = 20

    # NoSQL Database config
    nosql_url: str | None = None  # e.g., "mongodb://localhost:27017"
    nosql_database: str = "adaptive_nn"
    nosql_collection: str = "predictions"

    # General config
    enable_metrics: bool = True
    connection_timeout: int = 30


if SQLALCHEMY_AVAILABLE:
    class ModelPrediction(Base):
        """SQLAlchemy model for storing predictions."""
        __tablename__ = "model_predictions"

        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        model_name = Column(String(100), nullable=False)
        input_data = Column(JSON)
        predictions = Column(JSON)
        latency_ms = Column(Float)
        batch_size = Column(Integer)
        user_id = Column(String(100))
        session_id = Column(String(100))
        metadata = Column(JSON)


    class ModelMetrics(Base):
        """SQLAlchemy model for storing model metrics."""
        __tablename__ = "model_metrics"

        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        model_name = Column(String(100), nullable=False)
        metric_name = Column(String(100), nullable=False)
        metric_value = Column(Float)
        tags = Column(JSON)
else:
    class ModelPrediction:
        pass

    class ModelMetrics:
        pass


class DatabaseManager:
    """Abstract base class for database managers."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def store_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Store a prediction result."""
        raise NotImplementedError

    async def get_predictions(self, filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve predictions with optional filters."""
        raise NotImplementedError

    async def store_metrics(self, metrics: dict[str, float], model_name: str, tags: dict[str, str] | None = None) -> None:
        """Store model metrics."""
        raise NotImplementedError

    async def get_metrics(self, model_name: str, metric_names: list[str] | None = None,
                         start_time: datetime | None = None, end_time: datetime | None = None) -> list[dict[str, Any]]:
        """Retrieve model metrics."""
        raise NotImplementedError


class SQLManager(DatabaseManager):
    """SQL database manager using SQLAlchemy."""

    def __init__(self, config: DatabaseConfig):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy not available. Install with: pip install sqlalchemy")

        super().__init__(config)

        if not config.sql_url:
            raise ValueError("SQL URL not provided in config")

        self.engine = create_engine(
            config.sql_url,
            pool_size=config.sql_pool_size,
            max_overflow=config.sql_max_overflow,
            connect_args={"connect_timeout": config.connection_timeout}
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

    async def store_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Store a prediction result in SQL database."""
        try:
            with self.SessionLocal() as session:
                prediction = ModelPrediction(
                    model_name=prediction_data.get("model_name", "unknown"),
                    input_data=prediction_data.get("input_data"),
                    predictions=prediction_data.get("predictions"),
                    latency_ms=prediction_data.get("latency_ms"),
                    batch_size=prediction_data.get("batch_size"),
                    user_id=prediction_data.get("user_id"),
                    session_id=prediction_data.get("session_id"),
                    metadata=prediction_data.get("metadata", {})
                )

                session.add(prediction)
                session.commit()
                session.refresh(prediction)

                return str(prediction.id)

        except Exception as e:
            self.logger.error(f"Failed to store prediction: {e}")
            raise

    async def get_predictions(self, filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve predictions from SQL database."""
        try:
            with self.SessionLocal() as session:
                query = session.query(ModelPrediction)

                # Apply filters
                if filters:
                    if "model_name" in filters:
                        query = query.filter(ModelPrediction.model_name == filters["model_name"])
                    if "user_id" in filters:
                        query = query.filter(ModelPrediction.user_id == filters["user_id"])
                    if "start_time" in filters:
                        query = query.filter(ModelPrediction.timestamp >= filters["start_time"])
                    if "end_time" in filters:
                        query = query.filter(ModelPrediction.timestamp <= filters["end_time"])

                # Order by timestamp and limit
                predictions = query.order_by(ModelPrediction.timestamp.desc()).limit(limit).all()

                return [
                    {
                        "id": p.id,
                        "timestamp": p.timestamp.isoformat(),
                        "model_name": p.model_name,
                        "input_data": p.input_data,
                        "predictions": p.predictions,
                        "latency_ms": p.latency_ms,
                        "batch_size": p.batch_size,
                        "user_id": p.user_id,
                        "session_id": p.session_id,
                        "metadata": p.metadata
                    }
                    for p in predictions
                ]

        except Exception as e:
            self.logger.error(f"Failed to retrieve predictions: {e}")
            raise

    async def store_metrics(self, metrics: dict[str, float], model_name: str, tags: dict[str, str] | None = None) -> None:
        """Store model metrics in SQL database."""
        try:
            with self.SessionLocal() as session:
                metric_records = []

                for metric_name, metric_value in metrics.items():
                    metric_record = ModelMetrics(
                        model_name=model_name,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        tags=tags or {}
                    )
                    metric_records.append(metric_record)

                session.add_all(metric_records)
                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
            raise

    async def get_metrics(self, model_name: str, metric_names: list[str] | None = None,
                         start_time: datetime | None = None, end_time: datetime | None = None) -> list[dict[str, Any]]:
        """Retrieve model metrics from SQL database."""
        try:
            with self.SessionLocal() as session:
                query = session.query(ModelMetrics).filter(ModelMetrics.model_name == model_name)

                # Apply filters
                if metric_names:
                    query = query.filter(ModelMetrics.metric_name.in_(metric_names))
                if start_time:
                    query = query.filter(ModelMetrics.timestamp >= start_time)
                if end_time:
                    query = query.filter(ModelMetrics.timestamp <= end_time)

                metrics = query.order_by(ModelMetrics.timestamp.desc()).all()

                return [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "metric_name": m.metric_name,
                        "metric_value": m.metric_value,
                        "tags": m.tags
                    }
                    for m in metrics
                ]

        except Exception as e:
            self.logger.error(f"Failed to retrieve metrics: {e}")
            raise


class NoSQLManager(DatabaseManager):
    """NoSQL database manager using MongoDB."""

    def __init__(self, config: DatabaseConfig):
        if not PYMONGO_AVAILABLE:
            raise ImportError("PyMongo not available. Install with: pip install pymongo")

        super().__init__(config)

        if not config.nosql_url:
            raise ValueError("NoSQL URL not provided in config")

        self.client = MongoClient(
            config.nosql_url,
            serverSelectionTimeoutMS=config.connection_timeout * 1000
        )

        self.db = self.client[config.nosql_database]
        self.predictions_collection = self.db[config.nosql_collection]
        self.metrics_collection = self.db["model_metrics"]

        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # Predictions collection indexes
            self.predictions_collection.create_index([("timestamp", -1)])
            self.predictions_collection.create_index([("model_name", 1), ("timestamp", -1)])
            self.predictions_collection.create_index([("user_id", 1), ("timestamp", -1)])

            # Metrics collection indexes
            self.metrics_collection.create_index([("model_name", 1), ("timestamp", -1)])
            self.metrics_collection.create_index([("model_name", 1), ("metric_name", 1), ("timestamp", -1)])

        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {e}")

    async def store_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Store a prediction result in MongoDB."""
        try:
            # Add timestamp if not present
            if "timestamp" not in prediction_data:
                prediction_data["timestamp"] = datetime.utcnow()

            result = self.predictions_collection.insert_one(prediction_data)
            return str(result.inserted_id)

        except Exception as e:
            self.logger.error(f"Failed to store prediction: {e}")
            raise

    async def get_predictions(self, filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve predictions from MongoDB."""
        try:
            query = filters or {}

            cursor = self.predictions_collection.find(query).sort("timestamp", -1).limit(limit)
            predictions = []

            for doc in cursor:
                # Convert ObjectId to string
                doc["_id"] = str(doc["_id"])
                if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                predictions.append(doc)

            return predictions

        except Exception as e:
            self.logger.error(f"Failed to retrieve predictions: {e}")
            raise

    async def store_metrics(self, metrics: dict[str, float], model_name: str, tags: dict[str, str] | None = None) -> None:
        """Store model metrics in MongoDB."""
        try:
            timestamp = datetime.utcnow()

            metric_docs = []
            for metric_name, metric_value in metrics.items():
                doc = {
                    "timestamp": timestamp,
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "tags": tags or {}
                }
                metric_docs.append(doc)

            if metric_docs:
                self.metrics_collection.insert_many(metric_docs)

        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
            raise

    async def get_metrics(self, model_name: str, metric_names: list[str] | None = None,
                         start_time: datetime | None = None, end_time: datetime | None = None) -> list[dict[str, Any]]:
        """Retrieve model metrics from MongoDB."""
        try:
            query = {"model_name": model_name}

            # Apply filters
            if metric_names:
                query["metric_name"] = {"$in": metric_names}

            if start_time or end_time:
                time_query = {}
                if start_time:
                    time_query["$gte"] = start_time
                if end_time:
                    time_query["$lte"] = end_time
                query["timestamp"] = time_query

            cursor = self.metrics_collection.find(query).sort("timestamp", -1)
            metrics = []

            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                metrics.append(doc)

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to retrieve metrics: {e}")
            raise


class HybridDatabaseManager(DatabaseManager):
    """Hybrid database manager that uses both SQL and NoSQL."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)

        self.sql_manager = None
        self.nosql_manager = None

        # Initialize SQL manager if configured
        if config.sql_url:
            try:
                self.sql_manager = SQLManager(config)
                self.logger.info("SQL database manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SQL manager: {e}")

        # Initialize NoSQL manager if configured
        if config.nosql_url:
            try:
                self.nosql_manager = NoSQLManager(config)
                self.logger.info("NoSQL database manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NoSQL manager: {e}")

        if not self.sql_manager and not self.nosql_manager:
            raise ValueError("At least one database must be configured")

    async def store_prediction(self, prediction_data: dict[str, Any]) -> str:
        """Store prediction in both databases if available."""
        results = []

        # Store in SQL database (structured data)
        if self.sql_manager:
            try:
                result = await self.sql_manager.store_prediction(prediction_data)
                results.append(f"sql:{result}")
            except Exception as e:
                self.logger.error(f"SQL storage failed: {e}")

        # Store in NoSQL database (flexible schema)
        if self.nosql_manager:
            try:
                result = await self.nosql_manager.store_prediction(prediction_data)
                results.append(f"nosql:{result}")
            except Exception as e:
                self.logger.error(f"NoSQL storage failed: {e}")

        return ",".join(results) if results else "failed"

    async def get_predictions(self, filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve predictions from preferred database."""
        # Prefer NoSQL for flexible queries, fallback to SQL
        if self.nosql_manager:
            try:
                return await self.nosql_manager.get_predictions(filters, limit)
            except Exception as e:
                self.logger.error(f"NoSQL retrieval failed: {e}")

        if self.sql_manager:
            try:
                return await self.sql_manager.get_predictions(filters, limit)
            except Exception as e:
                self.logger.error(f"SQL retrieval failed: {e}")

        return []

    async def store_metrics(self, metrics: dict[str, float], model_name: str, tags: dict[str, str] | None = None) -> None:
        """Store metrics in both databases if available."""
        # Store in SQL database (good for time-series queries)
        if self.sql_manager:
            try:
                await self.sql_manager.store_metrics(metrics, model_name, tags)
            except Exception as e:
                self.logger.error(f"SQL metrics storage failed: {e}")

        # Store in NoSQL database (flexible schema)
        if self.nosql_manager:
            try:
                await self.nosql_manager.store_metrics(metrics, model_name, tags)
            except Exception as e:
                self.logger.error(f"NoSQL metrics storage failed: {e}")

    async def get_metrics(self, model_name: str, metric_names: list[str] | None = None,
                         start_time: datetime | None = None, end_time: datetime | None = None) -> list[dict[str, Any]]:
        """Retrieve metrics from preferred database."""
        # Prefer SQL for time-series queries, fallback to NoSQL
        if self.sql_manager:
            try:
                return await self.sql_manager.get_metrics(model_name, metric_names, start_time, end_time)
            except Exception as e:
                self.logger.error(f"SQL metrics retrieval failed: {e}")

        if self.nosql_manager:
            try:
                return await self.nosql_manager.get_metrics(model_name, metric_names, start_time, end_time)
            except Exception as e:
                self.logger.error(f"NoSQL metrics retrieval failed: {e}")

        return []
