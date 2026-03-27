"""Verdict cache — SQLite-backed caching for repeated claims.

Caches verification results by claim+context hash. Reduces cost and latency
for repeated or similar queries. Realistic 40-60% hit rate for support bots.

Usage:
    config = Config(cache_enabled=True, cache_ttl_seconds=3600)
    result = await verify("claim", config=config)  # First call: ~15s, costs ~$0.08
    result = await verify("claim", config=config)  # Cache hit: ~0ms, costs $0
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from veritas.core.result import VerificationResult


class VerdictCache:
    """SQLite-backed verdict cache with TTL expiry."""

    def __init__(self, db_path: str, ttl_seconds: int = 3600):
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self._ensure_db()

    def _ensure_db(self):
        """Create the cache database and table if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verdict_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON verdict_cache(created_at)
            """)

    @staticmethod
    def _make_key(claim: str, context: str | None, domain: str | None) -> str:
        """Create a deterministic cache key from claim + context + domain."""
        parts = [claim.strip().lower()]
        if context:
            parts.append(context.strip().lower())
        if domain:
            parts.append(domain.strip().lower())
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, claim: str, context: str | None = None, domain: str | None = None) -> VerificationResult | None:
        """Look up a cached verdict. Returns None on miss or expiry."""
        key = self._make_key(claim, context, domain)
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT result_json, created_at FROM verdict_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            result_json, created_at = row
            if now - created_at > self.ttl_seconds:
                conn.execute("DELETE FROM verdict_cache WHERE cache_key = ?", (key,))
                return None
            try:
                result = VerificationResult.model_validate_json(result_json)
                result.metadata["cache_hit"] = True
                return result
            except Exception:
                return None

    def put(self, claim: str, context: str | None, domain: str | None, result: VerificationResult) -> None:
        """Store a verdict in the cache."""
        key = self._make_key(claim, context, domain)
        result_json = result.model_dump_json()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO verdict_cache (cache_key, result_json, created_at) VALUES (?, ?, ?)",
                (key, result_json, time.time()),
            )

    def clear(self) -> int:
        """Clear all cached verdicts. Returns number of entries removed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM verdict_cache")
            return cursor.rowcount

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number removed."""
        cutoff = time.time() - self.ttl_seconds
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM verdict_cache WHERE created_at < ?", (cutoff,))
            return cursor.rowcount

    def stats(self) -> dict:
        """Return cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM verdict_cache").fetchone()[0]
            cutoff = time.time() - self.ttl_seconds
            valid = conn.execute("SELECT COUNT(*) FROM verdict_cache WHERE created_at >= ?", (cutoff,)).fetchone()[0]
            return {"total_entries": total, "valid_entries": valid, "expired_entries": total - valid}
