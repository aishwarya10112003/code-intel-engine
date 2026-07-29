"""
Package init. Loads secrets from a local `.env` file (if present) so you don't have to
`export` your API key every terminal session. The `.env` file is git-ignored.
"""
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not installed — fall back to real environment variables.
    pass
