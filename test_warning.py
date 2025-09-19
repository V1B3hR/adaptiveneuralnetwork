import logging
import sys
sys.path.insert(0, '.')

# Set up logging to capture warning messages
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

# Simulate the current warning message directly
logger.warning("No validation loader provided - evaluating on training set. "
               "This may lead to inflated accuracy estimates and mask overfitting!")

print("Current warning message displayed above.")
