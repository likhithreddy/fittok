"""Tests for the caching layer."""

from unittest.mock import patch, MagicMock

from fittok.cache import (
    _hash_str,
    cache_stats,
    clear_cache,
)


class TestHashStr:
    def test_deterministic(self):
        assert _hash_str("hello") == _hash_str("hello")

    def test_different_inputs(self):
        assert _hash_str("hello") != _hash_str("world")

    def test_length(self):
        assert len(_hash_str("test")) == 16


class TestCacheStats:
    @patch("fittok.cache._get_cache")
    def test_stats_no_cache(self, mock_get):
        mock_get.return_value = None
        stats = cache_stats()
        assert stats["available"] is False
        assert "stats" in stats

    @patch("fittok.cache._get_cache")
    def test_stats_with_cache(self, mock_get):
        mock_cache = MagicMock()
        mock_cache.__len__ = lambda self_inner: 5
        mock_cache.volume.return_value = 1024
        mock_get.return_value = mock_cache
        stats = cache_stats()
        assert stats["available"] is True


class TestClearCache:
    @patch("fittok.cache._get_cache")
    def test_clear_no_cache(self, mock_get):
        mock_get.return_value = None
        result = clear_cache()
        assert "error" in result
