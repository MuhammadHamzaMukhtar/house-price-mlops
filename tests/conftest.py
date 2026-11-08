# conftest.py - pytest runs this file first, before any test file.
# Send the API's prediction diary to a temporary file during tests,
# so test requests don't mix with real user traffic in predictions.csv.

import os
import tempfile

os.environ.setdefault("LOG_FILE", os.path.join(tempfile.mkdtemp(), "test_predictions.csv"))
