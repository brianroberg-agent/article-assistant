#!/usr/bin/env python3
"""
Unit tests for metadata extractors (NewAtlantisExtractor and LLMExtractor).
"""

import pytest
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup

from article_assistant import (
    NewAtlantisExtractor,
    LLMExtractor,
    get_extractor_for_url,
    ArticleMetadata,
    _edition_from_header,
    season_start_date,
)


class TestNewAtlantisExtractor:
    """Test cases for NewAtlantisExtractor."""

    def test_supports_url_thenewatlantis(self):
        """Test that NewAtlantisExtractor supports thenewatlantis.com URLs."""
        extractor = NewAtlantisExtractor()

        assert extractor.supports_url("https://www.thenewatlantis.com/article")
        assert extractor.supports_url("https://thenewatlantis.com/publications/test")
        assert not extractor.supports_url("https://example.com/article")

    def test_extract_metadata_json_ld(self):
        """Test extraction using JSON-LD (primary method)."""
        extractor = NewAtlantisExtractor()

        html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@type": "Article",
                "headline": "Test Article",
                "author": [{"name": "Author Name"}]
            }
            </script>
        </head>
        </html>
        """

        metadata = extractor.extract_metadata(html, "https://thenewatlantis.com/test")

        assert metadata["title"] == "Test Article"
        assert metadata["authors"] == ["Author Name"]
        assert metadata["publication"] == "The New Atlantis"

    def test_extract_metadata_with_issue_inference(self):
        """Test that edition number inference works."""
        extractor = NewAtlantisExtractor()

        html = """
        <html>
        <body>
            <h1>Test Article</h1>
            <p>Published in Winter 2025</p>
        </body>
        </html>
        """

        metadata = extractor.extract_metadata(html, "https://thenewatlantis.com/test")

        assert metadata["issue_number"] == "79"
        assert metadata["issue_season"] == "Winter 2025"


class TestLLMExtractor:
    """Test cases for LLMExtractor."""

    def test_llm_extractor_import_error(self):
        """Test that LLMExtractor raises ImportError when llm not installed."""
        with patch("article_assistant.llm", None):
            with pytest.raises(ImportError, match="llm package not installed"):
                LLMExtractor()

    @patch("article_assistant.llm")
    def test_llm_extractor_initialization(self, mock_llm):
        """Test LLMExtractor initialization with mocked llm."""
        mock_model = Mock()
        mock_llm.get_model.return_value = mock_model

        extractor = LLMExtractor(model_name="gpt-4o-mini")

        assert extractor.model == mock_model
        assert extractor.model_name == "gpt-4o-mini"
        mock_llm.get_model.assert_called_once_with("gpt-4o-mini")

    @patch("article_assistant.llm")
    def test_llm_extractor_supports_all_urls(self, mock_llm):
        """Test that LLMExtractor supports all URLs (fallback)."""
        mock_llm.get_model.return_value = Mock()
        extractor = LLMExtractor()

        assert extractor.supports_url("https://example.com/article")
        assert extractor.supports_url("https://medium.com/@user/post")
        assert extractor.supports_url("https://thenewatlantis.com/test")

    @patch("article_assistant.llm")
    def test_llm_extractor_extraction(self, mock_llm):
        """Test LLM extraction with mocked response."""
        # Mock LLM response
        mock_metadata = ArticleMetadata(
            title="Test Article",
            authors=["John Doe"],
            publication="Example Site",
            date_published="2025-01-15",
        )

        mock_response = Mock()
        mock_response.schema_obj.return_value = mock_metadata

        mock_model = Mock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        extractor = LLMExtractor()

        html = """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <h1>Test Article</h1>
            <p>By John Doe</p>
            <p>Published on Example Site</p>
        </body>
        </html>
        """

        metadata = extractor.extract_metadata(html, "https://example.com/test")

        assert metadata["title"] == "Test Article"
        assert metadata["authors"] == ["John Doe"]
        assert metadata["publication"] == "Example Site"
        assert metadata["date_published"] == "2025-01-15"

    @patch("article_assistant.llm")
    def test_llm_extractor_removes_script_tags(self, mock_llm):
        """Test that script and style tags are removed before LLM processing."""
        mock_metadata = ArticleMetadata(
            title="Clean Article", authors=[], publication="Test"
        )

        mock_response = Mock()
        mock_response.schema_obj.return_value = mock_metadata

        mock_model = Mock()
        mock_model.prompt.return_value = mock_response
        mock_llm.get_model.return_value = mock_model

        extractor = LLMExtractor()

        html = """
        <html>
        <head>
            <script>alert('test');</script>
            <style>.test { color: red; }</style>
        </head>
        <body>
            <h1>Clean Article</h1>
            <script>console.log('remove me');</script>
        </body>
        </html>
        """

        extractor.extract_metadata(html, "https://example.com/test")

        # Verify prompt doesn't contain script content
        call_args = mock_model.prompt.call_args[0][0]
        assert "alert('test')" not in call_args
        assert "console.log" not in call_args

    @patch("article_assistant.llm")
    def test_llm_extractor_error_handling(self, mock_llm):
        """Test error handling when LLM extraction fails."""
        mock_model = Mock()
        mock_model.prompt.side_effect = Exception("API Error")
        mock_llm.get_model.return_value = mock_model

        extractor = LLMExtractor()

        html = """
        <html>
        <head><title>Fallback Title</title></head>
        <body><p>Content</p></body>
        </html>
        """

        metadata = extractor.extract_metadata(html, "https://example.com/test")

        # Should return fallback metadata
        assert metadata["title"] == "Fallback Title"
        assert metadata["authors"] == []
        assert metadata["publication"] == "example.com"


class TestSiteDetection:
    """Test cases for get_extractor_for_url() routing."""

    def test_get_extractor_for_thenewatlantis(self):
        """Test that The New Atlantis URLs get NewAtlantisExtractor."""
        extractor = get_extractor_for_url("https://www.thenewatlantis.com/article")
        assert isinstance(extractor, NewAtlantisExtractor)

    @patch("article_assistant.llm")
    def test_get_extractor_for_generic_site(self, mock_llm):
        """Test that generic URLs get LLMExtractor."""
        mock_llm.get_model.return_value = Mock()

        extractor = get_extractor_for_url("https://example.com/article")
        assert isinstance(extractor, LLMExtractor)

    @patch("article_assistant.llm")
    def test_get_extractor_passes_model_name(self, mock_llm):
        """Test that model_name is passed to LLMExtractor."""
        mock_model = Mock()
        mock_llm.get_model.return_value = mock_model

        extractor = get_extractor_for_url(
            "https://example.com/article", model_name="claude-3.5-sonnet"
        )

        assert isinstance(extractor, LLMExtractor)
        assert extractor.model_name == "claude-3.5-sonnet"


GRAPH_PAGE = """
<html><head>
<meta name="description" content="Zoning made us lonely. The Court may end it.">
<meta property="og:description" content="Zoning made us lonely. The Court may end it.">
<script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "Article", "@id": "https://www.thenewatlantis.com/publications/zoning-alone#article",
   "author": {"@id": "https://www.thenewatlantis.com/authors/joseph-lawler#person"},
   "headline": "Zoning Alone", "datePublished": "2026-09-08T15:39:03+00:00"},
  {"@type": "WebPage", "@id": "https://www.thenewatlantis.com/publications/zoning-alone",
   "description": "Zoning made us lonely. The Court may end it."},
  {"@type": "Person", "@id": "https://www.thenewatlantis.com/authors/joseph-lawler#person",
   "name": "Joseph Lawler"}
]}
</script></head>
<body>
<nav><div class="hidden lg:block"><a class="link-transition" href="https://www.thenewatlantis.com/issues/no-87-winter-2027">No. 87 – Winter 2027</a></div></nav>
<div class="container">
  <div class="block text-center lg:hidden"><aside><p><a href="https://www.thenewatlantis.com/article_type/essay">Essay</a></p>
    <p><a class="link-transition" href="https://www.thenewatlantis.com/issues/no-86-fall-2026">Fall 2026</a></p></aside></div>
  <div class="text-center"><h1>Zoning Alone</h1></div>
</div>
<div class="container"><div class="gutenberg-content"><p>Body.</p></div></div>
<div class="container"><h5>Related</h5><article class="card"><p><a href="https://www.thenewatlantis.com/issues/no-23-winter-2009">Winter 2009</a></p></article></div>
</body></html>
"""
WEB_ONLY_PAGE = """
<html><body>
<nav><div class="hidden lg:block"><a class="link-transition" href="https://www.thenewatlantis.com/issues/no-86-fall-2026">No. 86 – Fall 2026</a></div></nav>
<div class="container">
  <div class="block text-center lg:hidden"><aside><p><a href="https://www.thenewatlantis.com/article_type/essay">Essay</a></p>
    <p>October 7, 2026</p></aside></div>
  <div class="text-center"><h1>Are We Really on the Path to AGI?</h1></div>
</div>
<div class="container"><h5>Related</h5><article class="card"><p><a href="https://www.thenewatlantis.com/issues/no-85-summer-2026">Summer 2026</a></p></article></div>
</body></html>
"""
WIDE_WRAPPER_PAGE = """
<html><body>
<main><div><h1>An Older Piece</h1></div>
  <div class="gutenberg-content"><p>Body.</p></div>
  <aside class="related"><a href="https://www.thenewatlantis.com/issues/no-85-summer-2026">Summer 2026</a></aside></main>
</body></html>
"""
URL = "https://www.thenewatlantis.com/publications/zoning-alone"


class TestNewAtlantisGraph:
    def test_reads_title_date_and_author_from_the_graph(self):
        md = NewAtlantisExtractor().extract_metadata(GRAPH_PAGE, URL)
        assert md["title"] == "Zoning Alone"
        assert md["date_published"] == "2026-09-08T15:39:03+00:00"
        assert md["authors"] == ["Joseph Lawler"]

    def test_reads_the_description(self):
        md = NewAtlantisExtractor().extract_metadata(GRAPH_PAGE, URL)
        assert md["description"] == "Zoning made us lonely. The Court may end it."

    def test_edition_comes_from_the_article_header_not_the_nav_or_related(self):
        # The nav names a different (current) issue and a Related card links another;
        # only the header aside's link is the article's own edition.
        md = NewAtlantisExtractor().extract_metadata(GRAPH_PAGE, URL)
        assert md["issue_number"] == "86"
        assert md["issue_season"] == "Fall 2026"

    def test_web_only_piece_has_no_edition(self):
        md = NewAtlantisExtractor().extract_metadata(WEB_ONLY_PAGE, URL)
        assert "issue_number" not in md and "issue_season" not in md

    def test_related_aside_under_a_wide_wrapper_is_not_the_header(self):
        # The h1's grandparent is the whole page, and the only aside below it
        # is a Related card for another issue. The header reader must not take
        # it; what the page-wide fallback then makes of "Summer 2026" is that
        # fallback's own, known limitation.
        assert (
            _edition_from_header(BeautifulSoup(WIDE_WRAPPER_PAGE, "html.parser"))
            is None
        )

    def test_parenthesised_issue_marker_still_works(self):
        page = (
            "<html><body><h1>T</h1><p>From No. 79 (Winter 2025) issue</p></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert (md["issue_number"], md["issue_season"]) == ("79", "Winter 2025")

    def test_date_falls_back_to_the_season_start(self):
        page = (
            "<html><body><div><div><h1>T</h1></div><aside>"
            '<a href="https://www.thenewatlantis.com/issues/no-86-fall-2026">Fall 2026</a>'
            "</aside></div></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["date_published"] == "2026-09-01"


class TestSeasonStartDate:
    def test_seasons(self):
        assert season_start_date("Spring", 2026) == "2026-03-01"
        assert season_start_date("Summer", 2026) == "2026-06-01"
        assert season_start_date("Fall", 2026) == "2026-09-01"
        assert season_start_date("Autumn", 2026) == "2026-09-01"
        # Winter YYYY is the first issue of year YYYY: meteorological winter
        # starts the December before.
        assert season_start_date("Winter", 2026) == "2025-12-01"

    def test_unknown(self):
        assert season_start_date("Monsoon", 2026) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
