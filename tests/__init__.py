import atexit
import os
import tempfile

# The unit tests run on the small TF-IDF sentiment model: it ships in git and
# loads in a second. The served transformer is not in git; its own tests load
# it explicitly and skip when it is not installed.
os.environ["DEFAKETIVE_SENTIMENT_MODEL"] = "tfidf"

# Accounts and the signing key go to a throwaway folder, so no test ever
# writes users or a secret key into the real instance/ folder.
_ACCOUNTS_FOLDER = tempfile.TemporaryDirectory(prefix="defaketive-test-accounts-")
atexit.register(_ACCOUNTS_FOLDER.cleanup)
os.environ["DEFAKETIVE_ACCOUNTS_DB"] = os.path.join(_ACCOUNTS_FOLDER.name, "accounts.db")
os.environ.setdefault("DEFAKETIVE_SECRET_KEY", "defaketive-test-secret")

# Analyses are kept on disk between runs of the app; the tests keep theirs in a
# throwaway folder too, never in instance/analysis_cache.
_CACHE_FOLDER = tempfile.TemporaryDirectory(prefix="defaketive-test-analysis-cache-")
atexit.register(_CACHE_FOLDER.cleanup)
os.environ["DEFAKETIVE_ANALYSIS_CACHE"] = _CACHE_FOLDER.name
