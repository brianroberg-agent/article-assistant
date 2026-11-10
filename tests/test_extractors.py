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
<html><head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "Article", "@id": "https://www.thenewatlantis.com/publications/agi#article",
   "author": {"@id": "https://www.thenewatlantis.com/authors/jane-roe#person"},
   "headline": "Are We Really on the Path to AGI?",
   "datePublished": "2026-10-07T12:00:00+00:00"},
  {"@type": "Person", "@id": "https://www.thenewatlantis.com/authors/jane-roe#person",
   "name": "Jane Roe"}
]}
</script></head>
<body>
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
ENTITY_PAGE = """
<html><head>
<script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "Article", "@id": "https://www.thenewatlantis.com/publications/notes#article",
   "author": {"@id": "https://www.thenewatlantis.com/authors/mark-oconnell#person"},
   "headline": "Notes &#038; Briefs &ldquo;Fall&rdquo;"},
  {"@type": "WebPage", "@id": "https://www.thenewatlantis.com/publications/notes",
   "description": "A roundup &#8212; of notes"},
  {"@type": "Person", "@id": "https://www.thenewatlantis.com/authors/mark-oconnell#person",
   "name": "Mark O&#8217;Connell"}
]}
</script></head>
<body><h1>Notes &amp; Briefs</h1></body></html>
"""
URL = "https://www.thenewatlantis.com/publications/zoning-alone"


def _page_with_issue_link(slug):
    """A minimal page whose header aside links /issues/<slug>."""
    return (
        "<html><body><div><div><h1>T</h1></div><aside>"
        f'<a href="https://www.thenewatlantis.com/issues/{slug}">label</a>'
        "</aside></div></body></html>"
    )


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

    def test_decodes_entities_in_the_graph_headline(self):
        # Yoast leaves HTML entities inside its JSON-LD strings and json.loads
        # does not decode them, so they would otherwise reach the YAML title.
        md = NewAtlantisExtractor().extract_metadata(ENTITY_PAGE, URL)
        assert md["title"] == "Notes & Briefs \u201cFall\u201d"

    def test_decodes_entities_in_a_graph_author_name(self):
        md = NewAtlantisExtractor().extract_metadata(ENTITY_PAGE, URL)
        assert md["authors"] == ["Mark O\u2019Connell"]

    def test_decodes_entities_in_the_webpage_description(self):
        md = NewAtlantisExtractor().extract_metadata(ENTITY_PAGE, URL)
        assert md["description"] == "A roundup \u2014 of notes"

    def test_meta_description_entities_are_decoded_by_the_parser(self):
        page = (
            '<html><head><meta property="og:description" '
            'content="Notes &amp; Briefs &#8212; the roundup"></head>'
            "<body><h1>T</h1></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["description"] == "Notes & Briefs \u2014 the roundup"

    @pytest.mark.parametrize(
        ("slug", "edition"),
        [
            ("no-38-winterspring-2013", ("38", "Winter/Spring 2013")),
            ("no-43-summerfall-2014", ("43", "Summer/Fall 2014")),
            ("no-49-springsummer-2016", ("49", "Spring/Summer 2016")),
            ("no-53-summerfall-2017", ("53", "Summer/Fall 2017")),
            ("no-56-summerfall-2018", ("56", "Summer/Fall 2018")),
        ],
    )
    def test_double_issue_slugs_give_both_seasons(self, slug, edition):
        soup = BeautifulSoup(_page_with_issue_link(slug), "html.parser")
        assert _edition_from_header(soup) == edition

    def test_cross_year_slug_still_gives_its_first_season(self):
        soup = BeautifulSoup(
            _page_with_issue_link("no-7-fall-2004-winter-2005"), "html.parser"
        )
        assert _edition_from_header(soup) == ("7", "Fall 2004")

    def test_double_issue_dates_from_its_first_season(self):
        page = _page_with_issue_link("no-43-summerfall-2014")
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["issue_number"] == "43"
        assert md["issue_season"] == "Summer/Fall 2014"
        assert md["date_published"] == "2014-06-01"

    def test_meta_description_beats_the_webpage_description(self):
        page = (
            '<html><head><meta name="description" content="Plain meta.">'
            '<script type="application/ld+json">'
            '{"@graph": [{"@type": "WebPage", "description": "Graph."}]}'
            "</script></head><body><h1>T</h1></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["description"] == "Plain meta."

    def test_plain_meta_description_wins_over_og_description(self):
        # The live web-only piece "Are We Really on the Path to AGI?" has a
        # byline as its og:description and the summary as its meta description.
        # The og tag comes first, so a first-in-document-order lookup fails.
        page = (
            '<html><head><meta property="og:description" content="By A. Writer">'
            '<meta name="description" content="The summary.">'
            "</head><body><h1>T</h1></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["description"] == "The summary."

    def test_og_description_is_used_when_the_plain_meta_is_absent(self):
        page = (
            '<html><head><meta property="og:description" content="Open graph.">'
            "</head><body><h1>T</h1></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["description"] == "Open graph."

    def test_webpage_description_is_used_when_both_metas_are_absent(self):
        page = (
            '<html><head><script type="application/ld+json">'
            '{"@graph": [{"@type": "WebPage", "description": "Graph."}]}'
            "</script></head><body><h1>T</h1></body></html>"
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["description"] == "Graph."

    def test_no_description_anywhere_leaves_the_key_out(self):
        page = "<html><body><h1>T</h1></body></html>"
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert "description" not in md

    def test_unresolvable_author_reference_falls_back_to_the_byline(self):
        page = (
            '<html><head><script type="application/ld+json">'
            '{"@graph": [{"@type": "Article", "headline": "T", '
            '"author": {"@id": "https://example.org/authors/x#person"}}]}'
            "</script></head>"
            '<body><h1>T</h1><p class="byline">By Jane Doe</p></body></html>'
        )
        md = NewAtlantisExtractor().extract_metadata(page, URL)
        assert md["authors"] == ["Jane Doe"]


# The visible byline, in the live pages' shape (2026-11-03): the title header is the
# h1's parent div.text-center, holding the title, the deck and, when the page shows
# one, the byline's author links; "Related" cards at the foot name other articles' authors.
def _byline_page(header_names, card_names=("Card Writer",), graph_author="A Coalition",
                 slot=True, header=True):
    links = " and ".join(
        f'<a href="https://www.thenewatlantis.com/authors/x{i}" class="link-author">{n}</a>'
        for i, n in enumerate(header_names))
    byline = (f'<div class="text-grey text-sm mt-3"><p class="font-callunasans italic">{links}</p></div>'
              if slot else "")
    title = (f'<div class="text-center"><h1>T</h1><div class="text-dark-grey">Deck.</div>{byline}</div>'
             if header else f"<div><h1>T</h1>{byline}</div>")
    cards = "".join(
        '<article class="card mb-8"><div class="mb-2 text-grey"><p class="font-callunasans italic">'
        f'<a href="https://www.thenewatlantis.com/authors/c" class="link-author">{n}</a></p></div></article>'
        for n in card_names)
    graph = ('<script type="application/ld+json">{"@graph": [{"@type": "Article", "headline": "T", '
             f'"author": {{"name": "{graph_author}"}}}}]}}</script>')
    return f"<html><head>{graph}</head><body>{title}<p>Body.</p>{cards}</body></html>"


class TestNewAtlantisByline:
    """kokoro-reader reads the byline aloud before the article (Brian, 2026-11-03):
    only the names the page shows under the title count as attributed."""

    def test_byline_is_the_names_under_the_title_not_the_related_cards(self):
        md = NewAtlantisExtractor().extract_metadata(
            _byline_page(["Ada Quill", "Mara O&#8217;Dell"]), URL)
        assert md["byline"] == ["Ada Quill", "Mara O\u2019Dell"]

    def test_byline_is_empty_when_only_the_structured_data_names_an_author(self):
        # The Pro-Human AI Declaration's shape: no byline under the title, an author in
        # the JSON-LD graph only, author links in the Related cards. authors keeps the
        # graph's name, as before.
        md = NewAtlantisExtractor().extract_metadata(_byline_page([], slot=False), URL)
        assert md["byline"] == []
        assert md["authors"] == ["A Coalition"]

    def test_byline_is_empty_for_an_empty_byline_and_no_cards(self):
        # Review finding: "400 Million Americans" has an empty byline under its title
        # and no Related cards, so no author link anywhere; its byline is still [].
        md = NewAtlantisExtractor().extract_metadata(_byline_page([], card_names=()), URL)
        assert md["byline"] == []

    def test_byline_is_left_out_when_the_page_has_no_title_header(self):
        # Not the markup this reads: the byline is unknown, not empty.
        md = NewAtlantisExtractor().extract_metadata(_byline_page(["Ada Quill"], header=False), URL)
        assert "byline" not in md

    def test_byline_names_are_trimmed_and_their_spaces_collapsed(self):
        md = NewAtlantisExtractor().extract_metadata(_byline_page(["  Ada \n  Quill "]), URL)
        assert md["byline"] == ["Ada Quill"]


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

    def test_double_season_uses_the_first(self):
        assert season_start_date("Summer/Fall", 2014) == "2014-06-01"
        assert season_start_date("Winter/Spring", 2013) == "2012-12-01"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# The print citation, in the live pages' shape (2026-11-10): a print article ends with
# a grey line citing its edition and pages, after the body and before the image
# credit. Web-only pieces and online-only interviews have none.
def _cited_page(citation, slug="no-86-fall-2026"):
    aside = (
        '<aside><p><a href="https://www.thenewatlantis.com/article_type/essay">Essay</a></p>'
        f'<p><a class="link-transition" href="https://www.thenewatlantis.com/issues/{slug}">'
        "label</a></p></aside>"
        if slug
        else "<aside><p>October 7, 2026</p></aside>"
    )
    cite = (
        f'<div class="font-callunasans text-grey text-sm mt-4">\n            {citation}'
        "          </div>"
        if citation
        else ""
    )
    return (
        "<html><body>"
        '<nav><a href="https://www.thenewatlantis.com/issues/no-87-winter-2027">No. 87 – Winter 2027</a></nav>'
        f'<div class="container"><div class="block text-center lg:hidden">{aside}</div>'
        '<div class="text-center"><h1>A Quiet Street</h1></div></div>'
        '<div class="container"><div class="gutenberg-content"><p>Body.</p></div>'
        f"{cite}"
        '<div class="font-callunasans text-grey text-sm mt-4">Header image: A. Photographer</div>'
        "</div></body></html>"
    )


CITATION = "Ada Quill, “A Quiet Street,” <em>The New Atlantis</em>, Number 86, Fall 2026, pp. 24–41."


class TestNewAtlantisPrintPages:
    """kokoro-reader puts the print pages in each episode's description (Brian,
    2026-11-10), when the page gives them."""

    def test_pages_come_from_the_print_citation(self):
        md = NewAtlantisExtractor().extract_metadata(_cited_page(CITATION), URL)
        assert md["pages"] == "pp. 24–41"

    def test_a_one_page_citation_gives_its_page(self):
        page = _cited_page(CITATION.replace("pp. 24–41", "p. 48"))
        assert NewAtlantisExtractor().extract_metadata(page, URL)["pages"] == "p. 48"

    def test_a_double_issue_citation_gives_its_pages(self):
        page = _cited_page(
            "Ada Quill, “A Quiet Street,” <em>The New Atlantis</em>, "
            "Number 43, Summer/Fall 2014, pp. 3–19.",
            slug="no-43-summerfall-2014",
        )
        assert NewAtlantisExtractor().extract_metadata(page, URL)["pages"] == "pp. 3–19"

    def test_a_citation_with_its_online_date_gives_its_pages(self):
        # A print article first published online cites both (review F1, live on
        # 2026-11-10): "…, pp. 115–121; TheNewAtlantis.com, April 16, 2026."
        page = _cited_page(
            CITATION.replace("pp. 24–41.", "pp. 24–41; TheNewAtlantis.com, April 16, 2026.")
        )
        assert NewAtlantisExtractor().extract_metadata(page, URL)["pages"] == "pp. 24–41"

    def test_a_page_without_a_citation_has_no_pages(self):
        md = NewAtlantisExtractor().extract_metadata(_cited_page(None), URL)
        assert md["issue_number"] == "86"
        assert "pages" not in md

    def test_a_citation_naming_another_issue_is_ignored(self):
        page = _cited_page(CITATION.replace("Number 86", "Number 85"))
        assert "pages" not in NewAtlantisExtractor().extract_metadata(page, URL)

    def test_a_piece_without_an_edition_has_no_pages(self):
        md = NewAtlantisExtractor().extract_metadata(_cited_page(CITATION, slug=None), URL)
        assert "issue_number" not in md
        assert "pages" not in md
