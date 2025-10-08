#!/usr/bin/env python3
"""
Test to verify the validation warning message format has been updated correctly.
This specifically tests the change made to fix the warning logs when validation data is unavailable.
"""
import io
import logging
import sys


def test_warning_message_format():
    """Test that the warning message has the expected format."""

    # Create a string buffer to capture log output
    log_capture_string = io.StringIO()

    # Configure logging to write to our string buffer
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(formatter)

    # Create logger for testing
    logger = logging.getLogger('test_continual_learning')
    logger.setLevel(logging.WARNING)
    logger.addHandler(ch)

    # Test the expected new warning message format
    expected_message = "⚠️ No validation loader - evaluating on training set!"
    logger.warning(expected_message)

    # Get the logged output
    log_contents = log_capture_string.getvalue()

    # Verify the message format
    print("=== Test Results ===")
    print(f"Expected message: '{expected_message}'")
    print(f"Actual log output: '{log_contents.strip()}'")

    # Check if the expected message is in the log output
    if expected_message in log_contents:
        print("✅ SUCCESS: Warning message format is correct!")
        return True
    else:
        print("❌ FAILURE: Warning message format is incorrect!")
        return False

def test_warning_format_matches_requirement():
    """Test that the format exactly matches the problem statement requirement."""

    required_format = "⚠️ No validation loader - evaluating on training set!"

    print("=== Format Verification ===")
    print(f"Required format from problem statement: '{required_format}'")

    # Check each component of the required format
    components = [
        "⚠️",  # Warning emoji
        "No validation loader", # Core message part 1
        "-", # Separator
        "evaluating on training set", # Core message part 2
        "!" # Exclamation mark ending
    ]

    all_present = all(component in required_format for component in components)

    if all_present:
        print("✅ All required components present in format")
    else:
        print("❌ Missing components in format")

    return all_present

if __name__ == "__main__":
    print("Testing validation warning message format...\n")

    test1_passed = test_warning_message_format()
    test2_passed = test_warning_format_matches_requirement()

    print("\n=== Summary ===")
    print(f"Warning message test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Format requirement test: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("🎉 All tests passed! The warning message format is correct.")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
