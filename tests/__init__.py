import logging

# The code under test logs freely; without a handler on the root logger Python prints WARNING+
# records to stderr ("last resort"), burying the test report. assertLogs still sees everything.
logging.getLogger().addHandler(logging.NullHandler())
