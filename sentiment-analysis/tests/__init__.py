"""Tests for the standalone sentiment analyzer."""

import os

# The unit tests run on the small TF-IDF sentiment model: it ships in git and
# loads in a second. The served transformer is not in git; its own tests load
# it explicitly and skip when it is not installed.
os.environ["DEFAKETIVE_SENTIMENT_MODEL"] = "tfidf"
