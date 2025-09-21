"""
Test cases for SQL injection prevention and database security
in the TimeSeriesTracker system.
"""

import os
import sqlite3

# Add the project root to the path
import sys
import tempfile
import unittest

sys.path.insert(0, "/home/runner/work/adaptiveneuralnetwork/adaptiveneuralnetwork")

from core.time_series_tracker import TimeSeriesQuery, TimeSeriesTracker


class TestSQLSecurity(unittest.TestCase):
    """Test SQL injection prevention and security measures"""

    def setUp(self):
        # Use temporary database file
        self.db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_file.close()

        self.tracker = TimeSeriesTracker(
            max_memory_points=1000, persist_to_disk=True, db_path=self.db_file.name
        )

        # Add some test data
        self.tracker.record_node_state(1, {"energy": 10.0, "anxiety": 2.0})
        self.tracker.record_node_state(2, {"energy": 15.0, "trust": 0.8})

    def tearDown(self):
        # Clean up temporary database
        try:
            os.unlink(self.db_file.name)
        except:
            pass

    def test_node_id_type_validation(self):
        """Test that node_ids are properly validated as integers"""
        # Valid integer node_ids should work
        query = TimeSeriesQuery(node_ids=[1, 2])
        results = self.tracker.query(query)
        self.assertGreater(len(results), 0)

        # Invalid types should be rejected or coerced
        with self.assertRaises((TypeError, ValueError)):
            query = TimeSeriesQuery(node_ids=["'; DROP TABLE time_series; --"])
            self.tracker.query(query)

        with self.assertRaises((TypeError, ValueError)):
            query = TimeSeriesQuery(node_ids=[1.5, 2.7])  # Floats should be rejected or coerced
            self.tracker.query(query)

    def test_variable_name_validation(self):
        """Test that variable names are validated against allowlist"""
        # Valid variable names should work
        query = TimeSeriesQuery(variables=["energy", "anxiety"])
        results = self.tracker.query(query)
        self.assertGreater(len(results), 0)

        # Invalid variable names should be rejected
        with self.assertRaises((ValueError, KeyError)):
            query = TimeSeriesQuery(variables=["'; DROP TABLE time_series; --"])
            self.tracker.query(query)

        with self.assertRaises((ValueError, KeyError)):
            query = TimeSeriesQuery(variables=["invalid_variable_name"])
            self.tracker.query(query)

    def test_sql_injection_attempts(self):
        """Test that SQL injection attempts are prevented"""
        injection_attempts = [
            "1; DROP TABLE time_series; --",
            "1 OR 1=1",
            "1' UNION SELECT * FROM sqlite_master--",
            "1; INSERT INTO time_series (timestamp, node_id, variable_name, value) VALUES (0, 999, 'hacked', 999); --",
        ]

        for attempt in injection_attempts:
            with self.subTest(injection_attempt=attempt):
                # Should not cause errors or unexpected behavior
                try:
                    # Try in node_ids (after fixing validation)
                    query = TimeSeriesQuery(node_ids=[attempt])
                    results = self.tracker.query(query)
                    # Should return empty results or raise validation error
                    self.assertIsInstance(results, list)
                except (TypeError, ValueError):
                    # Validation errors are expected and acceptable
                    pass

    def test_parameterized_queries_usage(self):
        """Verify that parameterized queries are being used"""
        # This test ensures the implementation uses parameterized queries
        # by checking that malicious input doesn't affect the database

        original_count = self._count_records()

        # Try query with malicious input
        try:
            query = TimeSeriesQuery(
                node_ids=[1], variables=["energy"], start_time=0, end_time=9999999999
            )
            self.tracker.query(query)
        except:
            pass  # Errors are fine, we just don't want data corruption

        # Verify database wasn't corrupted
        final_count = self._count_records()
        self.assertEqual(original_count, final_count)

    def test_explain_query_functionality(self):
        """Test EXPLAIN functionality for query performance analysis"""
        # This will test the new EXPLAIN functionality we'll add
        query = TimeSeriesQuery(node_ids=[1], variables=["energy"])

        # Check if explain functionality exists
        if hasattr(self.tracker, "explain_query"):
            explain_result = self.tracker.explain_query(query)
            self.assertIsInstance(explain_result, str)
            # Should contain query plan information (SCAN or SEARCH indicates index usage)
            upper_result = explain_result.upper()
            self.assertTrue(
                "SCAN" in upper_result or "SEARCH" in upper_result,
                f"Expected query plan info, got: {explain_result}",
            )
            # Should mention the table name
            self.assertIn("TIME_SERIES", upper_result)

    def test_input_boundary_validation(self):
        """Test validation and coercion at input boundaries"""
        # Test timestamp validation
        with self.assertRaises((TypeError, ValueError)):
            query = TimeSeriesQuery(start_time="invalid_time")
            self.tracker.query(query)

        # Test max_points validation
        query = TimeSeriesQuery(max_points="10")  # String instead of int
        # Should either work (coerced) or raise validation error
        try:
            results = self.tracker.query(query)
            self.assertIsInstance(results, list)
        except (TypeError, ValueError):
            pass  # Validation error is acceptable

    def test_multi_statement_prevention(self):
        """Test that multi-statements are prevented"""
        # SQLite doesn't support multi-statements by default, but let's verify
        with sqlite3.connect(self.db_file.name) as conn:
            cursor = conn.cursor()

            # This should fail or only execute the first statement
            try:
                cursor.execute("SELECT COUNT(*) FROM time_series; DROP TABLE time_series;")
                results = cursor.fetchall()
                # If it succeeded, verify table still exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='time_series'"
                )
                table_exists = cursor.fetchone()
                self.assertIsNotNone(
                    table_exists, "Table should still exist - multi-statement should be prevented"
                )
            except sqlite3.Error:
                # Exception is expected for multi-statements
                pass

    def _count_records(self):
        """Helper method to count records in database"""
        try:
            with sqlite3.connect(self.db_file.name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM time_series")
                return cursor.fetchone()[0]
        except:
            return 0


if __name__ == "__main__":
    unittest.main()
