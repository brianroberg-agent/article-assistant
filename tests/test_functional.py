#!/usr/bin/env python3
"""
Functional tests for article_assistant.py
Tests the script end-to-end with real and mock data.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import pytest
import article_assistant

from article_assistant import main
from tests.test_extractors import GRAPH_PAGE, URL, WEB_ONLY_PAGE


class TestFunctionalEndToEnd:
    """Functional tests for the complete script execution."""

    def test_script_network_error(self):
        """Test script behavior when network request fails."""
        result = subprocess.run(
            [
                sys.executable,
                "article_assistant.py",
                "metadata",
                "https://nonexistent-domain-12345.com/article",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 1
        assert "Error fetching article:" in result.stderr

    def test_script_no_arguments(self):
        """Test script behavior when no subcommand provided."""
        result = subprocess.run(
            [sys.executable, "article_assistant.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 2  # argparse error

    def test_script_help_message(self):
        """Test script help message display."""
        result = subprocess.run(
            [sys.executable, "article_assistant.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0
        assert "metadata" in result.stdout
        assert "content" in result.stdout


class TestFunctionalIntegration:
    """Integration tests using the main function directly."""

    @patch("article_assistant.requests.get")
    @patch("builtins.print")
    @patch("sys.argv", ["script.py", "metadata", "https://www.thenewatlantis.com/test"])
    def test_main_function_integration(self, mock_print, mock_get):
        """Test main function integration with mocked dependencies."""
        mock_html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@type": "Article",
                "headline": "Integration Test",
                "author": [{"name": "Integration Author"}]
            }
            </script>
        </head>
        </html>
        """

        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        main()

        # Verify print was called with the formatted output
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]

        assert "title: Integration Test" in output
        assert "  - Integration Author" in output
        assert "## Notes" in output

    @patch("article_assistant.requests.get")
    @patch(
        "sys.argv",
        ["script.py", "metadata", "--creation-date", "2023-05-01", "https://test.com"],
    )
    def test_main_function_with_custom_date(self, mock_get, capsys):
        """Test main function with custom creation date."""
        mock_html = """
        <html>
        <body><h1>Custom Date Test</h1></body>
        </html>
        """

        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        main()

        captured = capsys.readouterr()
        assert "creation-date: 2023-05-01" in captured.out

    @patch("article_assistant.llm")
    @patch("article_assistant.requests.get")
    @patch("sys.argv", ["script.py", "metadata", "https://example.com/article"])
    def test_main_function_url_validation_warning(self, mock_get, mock_llm, capsys):
        """Test main function with generic site (LLM-based extraction)."""
        mock_html = "<html><body><h1>Test</h1></body></html>"

        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock LLM to avoid actual API calls
        from article_assistant import ArticleMetadata

        mock_metadata = ArticleMetadata(
            title="Test Article", authors=["Test Author"], publication="Example.com"
        )
        mock_llm_response = Mock()
        mock_llm_response.schema_obj.return_value = mock_metadata
        mock_model = Mock()
        mock_model.prompt.return_value = mock_llm_response
        mock_llm.get_model.return_value = mock_model

        main()

        captured = capsys.readouterr()
        assert "Info: Using LLM-based extraction for this site" in captured.err


class TestFunctionalRealWorldHtml:
    """Tests with realistic HTML structures similar to The New Atlantis."""

    def test_complex_html_structure(self):
        """Test with complex HTML structure similar to real articles."""
        from article_assistant import extract_metadata, format_markdown_header

        complex_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>The New Atlantis - Complex Article</title>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": "The Future of Digital Humanities",
                "author": [
                    {"@type": "Person", "name": "Dr. Sarah  Johnson"},
                    {"@type": "Person", "name": "Prof. Michael   Chen"}
                ],
                "datePublished": "2025-02-01T00:00:00Z",
                "publisher": {
                    "@type": "Organization",
                    "name": "The New Atlantis"
                }
            }
            </script>
        </head>
        <body>
            <article>
                <header>
                    <h1>The Future of Digital Humanities</h1>
                    <div class="article-meta">
                        <span class="issue-info">No. 81 (Summer 2025)</span>
                        <span class="pages">pp. 15-32</span>
                    </div>
                </header>
                <div class="article-content">
                    <p>This article explores the intersection of technology and humanities...</p>
                </div>
            </article>
        </body>
        </html>
        """

        metadata = extract_metadata(complex_html)

        # Test that complex HTML is parsed correctly
        assert metadata["title"] == "The Future of Digital Humanities"
        assert len(metadata["authors"]) == 2
        assert "Dr. Sarah Johnson" in metadata["authors"]  # Whitespace normalized
        assert "Prof. Michael Chen" in metadata["authors"]  # Whitespace normalized
        assert metadata["issue_number"] == "81"
        assert metadata["issue_season"] == "Summer 2025"

        # Test formatted output
        formatted = format_markdown_header(metadata, "2025-06-19")

        assert "title: The Future of Digital Humanities" in formatted
        assert "  - Dr. Sarah Johnson" in formatted
        assert "  - Prof. Michael Chen" in formatted
        assert "periodical-edition: No. 81 (Summer 2025)" in formatted

    def test_fallback_html_parsing(self):
        """Test fallback HTML parsing when JSON-LD is malformed."""
        from article_assistant import extract_metadata

        fallback_html = """
        <html>
        <head>
            <title>The New Atlantis - Fallback Test</title>
            <script type="application/ld+json">
            {
                "malformed": "json"
                // This is invalid JSON
            }
            </script>
        </head>
        <body>
            <h1>Artificial Intelligence and Human Creativity</h1>
            <div class="byline">
                <span>By Dr. Emma Watson</span>
            </div>
            <div class="issue-details">
                Published in No. 82 (Fall 2025)
            </div>
        </body>
        </html>
        """

        metadata = extract_metadata(fallback_html)

        # Should fall back to HTML parsing
        assert metadata["title"] == "Artificial Intelligence and Human Creativity"
        assert metadata["authors"] == ["Dr. Emma Watson"]
        assert metadata["issue_number"] == "82"
        assert metadata["issue_season"] == "Fall 2025"

    def test_minimal_html_handling(self):
        """Test handling of minimal HTML with missing elements."""
        from article_assistant import extract_metadata, format_markdown_header

        minimal_html = """
        <html>
        <head><title>Minimal Article</title></head>
        <body>
            <p>Some content without proper structure</p>
        </body>
        </html>
        """

        metadata = extract_metadata(minimal_html)
        formatted = format_markdown_header(metadata, "2025-06-19")

        # Should handle gracefully with defaults
        assert metadata["title"] == "Minimal Article"
        assert metadata["publication"] == "The New Atlantis"
        assert "title: Minimal Article" in formatted
        assert "  - Unknown Author" in formatted  # Default author
        assert "publication: The New Atlantis" in formatted


def test_metadata_json_flag(capsys):
    with (
        patch.object(
            article_assistant, "fetch_article_content", return_value=GRAPH_PAGE
        ),
        patch.object(sys, "argv", ["article_assistant.py", "metadata", URL, "--json"]),
    ):
        article_assistant.main()
    out = json.loads(capsys.readouterr().out)
    assert out["title"] == "Zoning Alone"
    assert out["issue_number"] == "86"
    assert out["description"].startswith("Zoning made us lonely")


def test_metadata_json_carries_the_byline_and_the_yaml_header_does_not(capsys):
    # kokoro-reader reads "byline" from --json; the YAML header for Brian's notes keeps
    # its keys (REVIEW.md: a change aimed at --json can change his notes).
    page = (
        '<html><body><div class="text-center"><h1>T</h1><div class="text-grey">'
        '<p class="font-callunasans italic"><a href="https://www.thenewatlantis.com/authors/a" '
        'class="link-author">Ada Quill</a></p></div></div><p>Body.</p></body></html>'
    )
    with (
        patch.object(article_assistant, "fetch_article_content", return_value=page),
        patch.object(sys, "argv", ["article_assistant.py", "metadata", URL, "--json"]),
    ):
        article_assistant.main()
    assert json.loads(capsys.readouterr().out)["byline"] == ["Ada Quill"]
    with (
        patch.object(article_assistant, "fetch_article_content", return_value=page),
        patch.object(sys, "argv", ["article_assistant.py", "metadata", URL]),
    ):
        article_assistant.main()
    assert "byline" not in capsys.readouterr().out


def test_metadata_json_omits_the_edition_of_a_web_only_piece(capsys):
    # The pipeline reads a missing key as "absent": a web-only piece must not
    # carry an issue, but still carries the date and author from its page.
    with (
        patch.object(
            article_assistant, "fetch_article_content", return_value=WEB_ONLY_PAGE
        ),
        patch.object(sys, "argv", ["article_assistant.py", "metadata", URL, "--json"]),
    ):
        article_assistant.main()
    out = json.loads(capsys.readouterr().out)
    assert "issue_number" not in out
    assert "issue_season" not in out
    assert out["date_published"] == "2026-10-07T12:00:00+00:00"
    assert out["authors"] == ["Jane Roe"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
