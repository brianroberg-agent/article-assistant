#!/usr/bin/env python3
"""
Extract metadata from articles and generate Markdown headers for note-taking.
Supports The New Atlantis (specialized extraction) and generic sites (LLM-based extraction).
"""

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from pydantic import BaseModel, Field

# Optional llm import - gracefully handle if not installed
try:
    import llm
except ImportError:
    llm = None


class ArticleMetadata(BaseModel):
    """Pydantic schema for article metadata extraction with structured output."""

    title: str = Field(description="Article title or headline")
    authors: list[str] = Field(
        description="List of author names (full names, normalized)"
    )
    publication: str = Field(
        description="Publication name or website (e.g., 'The New Atlantis', 'Medium')"
    )
    date_published: Optional[str] = Field(
        default=None,
        description="Publication date in ISO format (YYYY-MM-DD) if available",
    )


class MetadataExtractor(ABC):
    """Abstract base class for metadata extractors."""

    @abstractmethod
    def extract_metadata(self, html_content: str, url: str) -> dict:
        """
        Extract metadata from HTML content.

        Args:
            html_content: Raw HTML content from the article page
            url: Original URL of the article (for context)

        Returns:
            Dictionary with keys: title, authors, publication,
            and optionally: date_published, issue_number, issue_season
        """
        pass

    @abstractmethod
    def supports_url(self, url: str) -> bool:
        """
        Check if this extractor supports the given URL.

        Args:
            url: The URL to check

        Returns:
            True if this extractor can handle the URL, False otherwise
        """
        pass

    @abstractmethod
    def extract_content(
        self, html_content: str, url: str, include_images: bool = True
    ) -> str:
        """
        Extract article body as Markdown.

        Args:
            html_content: Raw HTML content from the article page
            url: Original URL of the article (for context)
            include_images: If True, render images as ![alt](url). If False, strip them.

        Returns:
            Article body as a Markdown string.

        Raises:
            ValueError: If the article body cannot be located.
        """
        pass


def fetch_article_content(url):
    """Fetch the HTML content of the article."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching article: {e}", file=sys.stderr)
        sys.exit(1)


def infer_edition_number(season, year):
    """
    Infer The New Atlantis edition number from season and year.

    Based on the quarterly publication schedule:
    - Winter 2025 = No. 79
    - Summer 2025 = No. 81
    """
    if not season or not year:
        return None

    try:
        year = int(year)
    except (ValueError, TypeError):
        return None

    season_lower = season.lower()

    # Base calculation: Winter 2025 = 79
    # Each year has 4 issues, so year difference * 4
    base_year = 2025
    base_winter_number = 79

    # Calculate base number for the given year's winter issue
    year_diff = year - base_year
    year_base_number = base_winter_number + (year_diff * 4)

    # Season offsets within a year (Winter = 0, Spring = 1, Summer = 2, Fall = 3)
    season_offset = {"winter": 0, "spring": 1, "summer": 2, "fall": 3, "autumn": 3}

    offset = season_offset.get(season_lower)
    if offset is None:
        return None

    return year_base_number + offset


SEASON_START = {
    "spring": (0, "03-01"),
    "summer": (0, "06-01"),
    "fall": (0, "09-01"),
    "autumn": (0, "09-01"),
    "winter": (-1, "12-01"),
}


def season_start_date(season, year):
    """First day of an edition's meteorological season, as YYYY-MM-DD.

    "Winter 2026" is the first issue of 2026 (see infer_edition_number), and
    meteorological winter starts the previous December, so it maps to
    2025-12-01. Returns None for an unknown season.
    """
    entry = SEASON_START.get(str(season).lower())
    if entry is None:
        return None
    year_offset, month_day = entry
    return f"{int(year) + year_offset}-{month_day}"


def _jsonld_nodes(data):
    """Flatten a JSON-LD payload into its nodes (handles @graph and lists)."""
    if isinstance(data, list):
        return [n for n in data if isinstance(n, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("@graph"), list):
            return [n for n in data["@graph"] if isinstance(n, dict)]
        return [data]
    return []


def _has_type(node, wanted):
    t = node.get("@type")
    return t == wanted or (isinstance(t, list) and wanted in t)


ISSUE_SLUG_RE = re.compile(
    r"/issues/no-(\d+)-(winter|spring|summer|fall|autumn)-(\d{4})", re.I
)


def _edition_from_header(soup):
    """The article's own edition, read from the header aside beside its h1.

    On today's layout the aside sits in the block just before the h1's own
    block (``div > aside``, then ``div > h1``); it may also be that block's
    direct neighbour. Only those two places count: a descendant search from
    higher up would take the first aside anywhere below, such as a "Related"
    card for another issue.

    Returns (number, "Season Year") when the aside links an issue, "none" when
    the aside exists but links no issue (a web-only piece), and None when the
    page has no such header (older layouts), so callers may fall back.
    """
    h1 = soup.find("h1")
    if h1 is None or h1.parent is None:
        return None
    holder = h1.parent
    aside = None
    for block in (holder.find_previous_sibling(True), holder.find_next_sibling(True)):
        if block is None:
            continue
        aside = block if block.name == "aside" else block.find("aside", recursive=False)
        if aside is not None:
            break
    if aside is None:
        return None
    for link in aside.select('a[href*="/issues/"]'):
        m = ISSUE_SLUG_RE.search(link.get("href", ""))
        if m:
            return m.group(1), f"{m.group(2).title()} {m.group(3)}"
    return "none"


class NewAtlantisExtractor(MetadataExtractor):
    """
    Specialized extractor for The New Atlantis articles.

    Uses JSON-LD structured data with HTML fallback, and infers
    edition numbers from season/year information.
    """

    def supports_url(self, url: str) -> bool:
        """Check if URL is from The New Atlantis."""
        parsed = urlparse(url)
        return "thenewatlantis.com" in parsed.netloc

    def extract_metadata(self, html_content: str, url: str) -> dict:
        """
        Extract metadata using The New Atlantis-specific logic.

        This is the existing extract_metadata() function logic,
        moved into the class.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        metadata = {}

        # Try to extract from JSON-LD structured data first
        nodes = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                nodes.extend(_jsonld_nodes(json.loads(script.string or "")))
            except json.JSONDecodeError:
                continue
        by_id = {n["@id"]: n for n in nodes if isinstance(n.get("@id"), str)}

        article = next((n for n in nodes if _has_type(n, "Article")), None)
        if article:
            metadata["title"] = (article.get("headline") or "").strip()
            authors = article.get("author", [])
            if isinstance(authors, dict):
                authors = [authors]
            names = []
            for author in authors:
                if not isinstance(author, dict):
                    continue
                if not author.get("name") and isinstance(author.get("@id"), str):
                    author = by_id.get(author["@id"], author)
                if author.get("name"):
                    names.append(re.sub(r"\s+", " ", author["name"].strip()))
            if names:
                metadata["authors"] = names
            if article.get("datePublished"):
                metadata["date_published"] = article["datePublished"]

        description = None
        for attrs in ({"property": "og:description"}, {"name": "description"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content", "").strip():
                description = tag["content"].strip()
                break
        if description is None:
            page = next((n for n in nodes if _has_type(n, "WebPage")), None)
            if page and page.get("description"):
                description = page["description"].strip()
        if description:
            metadata["description"] = description

        # Fallback to HTML parsing if JSON-LD doesn't work
        if not metadata.get("title"):
            title_tag = soup.find("h1") or soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()

        # Extract author from byline if not found in JSON-LD
        if not metadata.get("authors"):
            byline = soup.find(class_=re.compile(r"author|byline", re.I))
            if byline:
                author_text = byline.get_text().strip()
                # Clean up common prefixes
                author_text = re.sub(r"^(by|author:?)\s*", "", author_text, flags=re.I)
                # Normalize whitespace
                author_text = re.sub(r"\s+", " ", author_text)
                metadata["authors"] = [author_text]

        # The article's own edition is the link in its header aside. The
        # page-wide search below runs only for pages without that header: the
        # nav and Related cards also name other issues.
        edition = _edition_from_header(soup)
        if isinstance(edition, tuple):
            metadata["issue_number"], metadata["issue_season"] = edition

        if edition is None:
            # Extract issue information
            issue_info = soup.find(string=re.compile(r"No\.\s*\d+"))
            if issue_info:
                issue_match = re.search(r"No\.\s*(\d+)\s*\(([^)]+)\)", issue_info)
                if issue_match:
                    metadata["issue_number"] = issue_match.group(1)
                    metadata["issue_season"] = issue_match.group(2)

            # If no explicit issue number found, look for season-only patterns
            if not metadata.get("issue_number"):
                # Look for patterns like "Winter 2025", "Spring 2024", etc.
                season_patterns = [
                    r"(Winter|Spring|Summer|Fall|Autumn)\s+(\d{4})",
                    r"(\d{4})\s+(Winter|Spring|Summer|Fall|Autumn)",
                ]

                for pattern in season_patterns:
                    season_match = soup.find(string=re.compile(pattern, re.IGNORECASE))
                    if season_match:
                        match = re.search(pattern, season_match, re.IGNORECASE)
                        if match:
                            if match.group(1).isdigit():
                                # Pattern: "2025 Winter"
                                year, season = match.group(1), match.group(2)
                            else:
                                # Pattern: "Winter 2025"
                                season, year = match.group(1), match.group(2)

                            # Infer the issue number
                            inferred_number = infer_edition_number(season, int(year))
                            if inferred_number:
                                metadata["issue_number"] = str(inferred_number)
                                metadata["issue_season"] = f"{season} {year}"
                            break

        # No datePublished: use the start of the edition's season
        if not metadata.get("date_published") and metadata.get("issue_season"):
            parts = metadata["issue_season"].split()
            if len(parts) == 2 and parts[1].isdigit():
                fallback = season_start_date(parts[0], int(parts[1]))
                if fallback:
                    metadata["date_published"] = fallback

        # Set publication name
        metadata["publication"] = "The New Atlantis"

        return metadata

    def extract_content(
        self, html_content: str, url: str, include_images: bool = True
    ) -> str:
        """Extract article body as Markdown using site-specific selectors."""
        soup = BeautifulSoup(html_content, "html.parser")

        body = (
            soup.find("div", class_="gutenberg-content")
            or soup.find("article")
            or soup.find("main")
            or soup.find("body")
        )

        if not body or not body.get_text(strip=True):
            raise ValueError("Could not locate article body")

        # Strip noise elements within the body
        for tag in body.find_all(["script", "style", "nav", "aside"]):
            tag.decompose()
        # Strip tooltip/paywall prompts
        for tag in body.find_all("div", class_="tooltip-container"):
            tag.decompose()
        # Strip promotional "Keep reading" epigraph blocks at end
        for tag in body.find_all(
            "div", class_=lambda c: c and "wp-block-lazyblock-epigraph" in c
        ):
            tag.decompose()

        strip_tags = ["img"] if not include_images else []
        md = markdownify(str(body), heading_style="ATX", strip=strip_tags)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()


class LLMExtractor(MetadataExtractor):
    """
    Generic extractor using LLM for metadata extraction.

    Uses Simon Willison's llm utility with structured output (Pydantic schemas)
    to extract metadata from arbitrary websites.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize LLM extractor.

        Args:
            model_name: Name of the LLM model to use (default: gpt-4o-mini)

        Raises:
            ImportError: If llm package is not installed
            RuntimeError: If llm model is not available or API key not configured
        """
        if llm is None:
            raise ImportError(
                "llm package not installed. Install with: pip install llm\n"
                "Then configure API keys with: llm keys set openai"
            )

        try:
            self.model = llm.get_model(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize LLM model '{model_name}': {e}\n"
                f"Ensure you have configured API keys: llm keys set openai"
            ) from e

        self.model_name = model_name

    def supports_url(self, url: str) -> bool:
        """LLMExtractor supports all URLs (fallback extractor)."""
        return True

    def extract_metadata(self, html_content: str, url: str) -> dict:
        """
        Extract metadata using LLM with structured output.

        Args:
            html_content: Raw HTML content
            url: Original URL (included in context for LLM)

        Returns:
            Dictionary with extracted metadata
        """
        # Convert HTML to text
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract text and limit length for token efficiency
        text = soup.get_text(separator="\n", strip=True)
        # Limit to ~4000 characters (roughly 1000 tokens)
        text = text[:4000]

        # Create prompt with context
        prompt = f"""Extract metadata from this article webpage.

URL: {url}

Article content:
{text}

Extract the following information:
- title: The article title or headline
- authors: List of author names (if available, otherwise empty list)
- publication: Name of the website or publication
- date_published: Publication date in YYYY-MM-DD format if available

Be accurate and only extract information that is clearly present."""

        try:
            # Use structured output for reliable extraction
            response = self.model.prompt(
                prompt,
                system="You are a metadata extraction assistant. Extract article metadata accurately and return structured data. If information is not available, use empty strings or empty lists as appropriate.",
                schema=ArticleMetadata,
            )

            metadata_obj = response.schema_obj()

            # Convert Pydantic model to dict format matching NewAtlantisExtractor
            return {
                "title": metadata_obj.title,
                "authors": metadata_obj.authors,
                "publication": metadata_obj.publication,
                "date_published": metadata_obj.date_published,
            }

        except Exception as e:
            print(f"Error during LLM extraction: {e}", file=sys.stderr)
            # Return minimal metadata on failure
            return {
                "title": soup.find("title").get_text()
                if soup.find("title")
                else "Unknown Title",
                "authors": [],
                "publication": urlparse(url).netloc,
            }

    def extract_content(
        self, html_content: str, url: str, include_images: bool = True
    ) -> str:
        """Extract article body as Markdown using markdownify with LLM fallback."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Find article body element
        body = soup.find("article") or soup.find("main") or soup.find("body")

        md_result = ""
        if body:
            for tag in body.find_all(
                ["script", "style", "nav", "header", "footer", "aside"]
            ):
                tag.decompose()

            strip_tags = ["img"] if not include_images else []
            md_result = markdownify(str(body), heading_style="ATX", strip=strip_tags)
            md_result = re.sub(r"\n{3,}", "\n\n", md_result).strip()

            if len(md_result) > 200:
                return md_result

        # Fallback to LLM for thin or missing markdownify results
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = text[:20000]

        image_instruction = (
            "Include images as ![alt text](url)."
            if include_images
            else "Do not include any images."
        )

        prompt = f"""Render the main article body from the following webpage as clean Markdown.

URL: {url}

- Preserve all headings, paragraphs, lists, blockquotes, and emphasis.
- {image_instruction}
- Exclude navigation, ads, author bios, and related article links.
- Output ONLY the article body. No front matter or commentary.

Content:
{text}"""

        try:
            response = self.model.prompt(
                prompt,
                system="You are a content extraction assistant. "
                "Convert article content to clean, well-structured Markdown.",
            )
            return response.text()
        except Exception as e:
            print(f"Error during LLM content extraction: {e}", file=sys.stderr)
            if md_result:
                return md_result
            raise ValueError(f"Could not extract article content: {e}") from e


def get_extractor_for_url(
    url: str, model_name: str = "gpt-4o-mini"
) -> MetadataExtractor:
    """
    Return appropriate metadata extractor based on URL.

    Args:
        url: The article URL to extract from
        model_name: LLM model name for generic extraction (default: gpt-4o-mini)

    Returns:
        MetadataExtractor instance (NewAtlantisExtractor or LLMExtractor)

    Raises:
        ImportError: If LLMExtractor is needed but llm is not installed
    """
    parsed_url = urlparse(url)

    # Check specialized extractors first
    if "thenewatlantis.com" in parsed_url.netloc:
        return NewAtlantisExtractor()

    # Fall back to LLM for generic sites
    return LLMExtractor(model_name=model_name)


def extract_metadata(html_content):
    """
    Extract metadata from The New Atlantis article HTML.

    DEPRECATED: This function is kept for backward compatibility with existing tests.
    New code should use NewAtlantisExtractor directly.
    """
    extractor = NewAtlantisExtractor()
    return extractor.extract_metadata(html_content, "")


def format_markdown_header(metadata, creation_date=None):
    """Format the extracted metadata into the required Markdown header."""
    if creation_date is None:
        creation_date = datetime.now().strftime("%Y-%m-%d")

    title = metadata.get("title", "Unknown Title")
    authors = metadata.get("authors", ["Unknown Author"])
    publication = metadata.get("publication", "The New Atlantis")

    # Format periodical edition
    periodical_edition = ""
    if metadata.get("issue_number") and metadata.get("issue_season"):
        periodical_edition = (
            f"No. {metadata['issue_number']} ({metadata['issue_season']})"
        )

    # Build the YAML front matter
    yaml_lines = ["---", f"title: {title}", "author:"]

    for author in authors:
        yaml_lines.append(f"  - {author}")

    yaml_lines.extend(
        [
            "format: journal article",
            f"creation-date: {creation_date}",
            f"publication: {publication}",
        ]
    )

    if periodical_edition:
        yaml_lines.append(f"periodical-edition: {periodical_edition}")

    yaml_lines.extend(["---", "", "## Notes"])

    return "\n".join(yaml_lines)


def _run_metadata(args):
    """Handle the metadata subcommand."""
    html_content = fetch_article_content(args.url)

    try:
        extractor = get_extractor_for_url(args.url, model_name=args.model)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nFor generic site extraction, install: uv add llm", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    metadata = extractor.extract_metadata(html_content, args.url)

    extractor_name = extractor.__class__.__name__
    if extractor_name != "NewAtlantisExtractor":
        print(f"# Using {extractor_name} with model: {args.model}", file=sys.stderr)

    parsed_url = urlparse(args.url)
    if "thenewatlantis.com" not in parsed_url.netloc:
        print("Info: Using LLM-based extraction for this site", file=sys.stderr)

    markdown_header = format_markdown_header(metadata, args.creation_date)
    print(markdown_header)


def _run_content(args):
    """Handle the content subcommand."""
    html_content = fetch_article_content(args.url)

    try:
        extractor = get_extractor_for_url(args.url, model_name=args.model)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    include_images = not args.no_images
    try:
        content = extractor.extract_content(html_content, args.url, include_images)
    except ValueError as e:
        print(f"Error extracting content: {e}", file=sys.stderr)
        sys.exit(1)

    print(content)


def main():
    parser = argparse.ArgumentParser(
        description="Article Assistant: extract metadata or content from articles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # metadata subcommand
    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Extract article metadata as YAML front matter",
    )
    metadata_parser.add_argument("url", help="URL of the article")
    metadata_parser.add_argument(
        "--creation-date",
        help="Override creation date (YYYY-MM-DD format)",
        default=None,
    )
    metadata_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for generic extraction (default: gpt-4o-mini). "
        "Only used for non-specialized sites.",
    )
    # content subcommand
    content_parser = subparsers.add_parser(
        "content",
        help="Extract article body as Markdown",
    )
    content_parser.add_argument("url", help="URL of the article")
    content_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for generic extraction (default: gpt-4o-mini). "
        "Only used for non-specialized sites.",
    )
    content_parser.add_argument(
        "--no-images",
        action="store_true",
        default=False,
        help="Strip images from the output",
    )
    args = parser.parse_args()

    if args.command == "metadata":
        _run_metadata(args)
    elif args.command == "content":
        _run_content(args)


if __name__ == "__main__":
    main()
