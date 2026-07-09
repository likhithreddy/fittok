"""Tests for the PII scrubbing module."""

from pathlib import Path

from fittok.pii_scrubber import (
    scrub_text,
    scrub_file,
    list_pii_patterns,
    add_pii_pattern,
)


class TestScrubText:
    def test_detect_email(self):
        result = scrub_text("Contact user@example.com for help")
        assert result["count"] >= 1
        assert "[REDACTED_EMAIL]" in result["scrubbed"]
        assert "user@example.com" not in result["scrubbed"]

    def test_detect_aws_key(self):
        result = scrub_text("Key: AKIAIOSFODNN7EXAMPLE")
        assert result["count"] >= 1
        assert "[REDACTED_AWS_ACCESS_KEY]" in result["scrubbed"]

    def test_detect_bearer_token(self):
        result = scrub_text("Authorization: Bearer abc123def456")
        assert result["count"] >= 1
        assert "[REDACTED_BEARER_TOKEN]" in result["scrubbed"]

    def test_detect_private_key(self):
        result = scrub_text("-----BEGIN RSA PRIVATE KEY-----\nblah\n-----END RSA PRIVATE KEY-----")
        assert result["count"] >= 1
        assert "[REDACTED_PRIVATE_KEY]" in result["scrubbed"]

    def test_detect_ip(self):
        result = scrub_text("Server at 192.168.1.100 is down")
        assert result["count"] >= 1
        assert "[REDACTED_IP_ADDRESS]" in result["scrubbed"]

    def test_detect_github_token(self):
        result = scrub_text("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert result["count"] >= 1

    def test_no_pii(self):
        result = scrub_text("Hello world, nothing to see here")
        assert result["count"] == 0
        assert result["scrubbed"] == "Hello world, nothing to see here"

    def test_custom_patterns(self):
        result = scrub_text(
            "My Slack token is xoxb-123-456",
            custom_patterns={"slack_token": r"xoxb-[0-9\-]+"},
        )
        assert result["count"] >= 1
        assert "[REDACTED_SLACK_TOKEN]" in result["scrubbed"]

    def test_findings_have_details(self):
        result = scrub_text("Email: test@test.com")
        assert len(result["findings"]) >= 1
        f = result["findings"][0]
        assert "type" in f
        assert "start" in f
        assert "end" in f


class TestScrubFile:
    def test_scrub_file(self, tmp_path):
        (tmp_path / "secret.py").write_text(
            'API_KEY = "sk-1234567890abcdefghijklmnop"\n'
        )
        result = scrub_file(str(tmp_path / "secret.py"))
        assert result["count"] >= 1
        assert "output_path" in result
        # Default output lands in the cache dir, NOT next to the source file.
        from fittok.cache import CACHE_DIR
        assert result["output_path"].startswith(CACHE_DIR)
        assert Path(result["output_path"]).exists()
        assert not (tmp_path / "secret.py.scrubbed").exists()

    def test_scrub_file_custom_output(self, tmp_path):
        (tmp_path / "data.txt").write_text("user@example.com")
        out = str(tmp_path / "clean.txt")
        result = scrub_file(str(tmp_path / "data.txt"), output_path=out)
        assert result["output_path"] == out
        assert Path(out).exists()

    def test_scrub_file_missing(self):
        result = scrub_file("/nonexistent/file.txt")
        assert "error" in result


class TestListPatterns:
    def test_list(self):
        result = list_pii_patterns()
        assert result["count"] >= 8
        assert "email" in result["patterns"]
        assert "aws_access_key" in result["patterns"]


class TestAddPattern:
    def test_add_valid(self):
        result = add_pii_pattern("test_pattern", r"TEST-\d+")
        assert result["added"] is True
        # Verify it works
        scrubbed = scrub_text("Found TEST-123 here")
        assert any(f["type"] == "test_pattern" for f in scrubbed["findings"])

    def test_add_invalid_regex(self):
        result = add_pii_pattern("bad", r"[invalid(")
        assert "error" in result
