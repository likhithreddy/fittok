"""Tests for the LLMLingua wrapper module."""

from unittest.mock import patch, MagicMock

from fittok.llmlingua_wrapper import compress_context


class TestCompressContext:
    def test_empty_input(self):
        result = compress_context("", "any question")
        assert result["compressed"] == ""
        assert result["original_tokens"] == 0
        assert result["compression_ratio"] == 0.0

    def test_already_within_budget(self):
        text = "Hello world"
        result = compress_context(text, "greeting", target_tokens=1000)
        assert result["compressed"] == text
        assert result["compression_ratio"] == 1.0

    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_compression_with_mock(self, mock_get):
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {
            "compressed_prompt": "compressed output"
        }
        mock_get.return_value = mock_compressor

        long_text = "word " * 500
        result = compress_context(long_text, "summary", target_tokens=50)

        assert result["compressed"] == "compressed output"
        assert result["original_tokens"] > 50
        assert mock_compressor.compress_prompt.called

    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_compression_list_result(self, mock_get):
        """LLMLingua may return a list of strings."""
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {
            "compressed_prompt": ["part one", "part two"]
        }
        mock_get.return_value = mock_compressor

        long_text = "word " * 500
        result = compress_context(long_text, "summary", target_tokens=50)

        assert result["compressed"] == "part one part two"

    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_compression_failure_fallback(self, mock_get):
        """Should fall back to truncation on failure."""
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.side_effect = RuntimeError("model error")
        mock_get.return_value = mock_compressor

        long_text = "word " * 500
        result = compress_context(long_text, "summary", target_tokens=50)

        assert len(result["compressed"]) > 0
        assert result["compressed_tokens"] > 0

    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_rate_override(self, mock_get):
        """When rate is provided, it should be used instead of computing from target_tokens."""
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {
            "compressed_prompt": "short"
        }
        mock_get.return_value = mock_compressor

        text = "word " * 200
        compress_context(text, "q", target_tokens=10, rate=0.5)
        call_kwargs = mock_compressor.compress_prompt.call_args
        assert call_kwargs[1]["rate"] == 0.5

    @patch("fittok.llmlingua_wrapper._get_compressor")
    def test_model_param_forwarded(self, mock_get):
        """Model parameter should be forwarded to _get_compressor."""
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt.return_value = {
            "compressed_prompt": "ok"
        }
        mock_get.return_value = mock_compressor

        compress_context("word " * 200, "q", target_tokens=50, model="custom-model")
        mock_get.assert_called_with("custom-model", None)
