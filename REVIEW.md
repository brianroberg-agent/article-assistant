# Reviewing article-assistant

## What this repo is

One script, `article_assistant.py`, with two subcommands:

- `metadata URL` prints a YAML header (title, author list, format, creation date, publication, periodical edition) for a Markdown note. With `--json` it prints the extracted fields as one JSON object instead.
- `content URL` prints the article's text as Markdown. `--no-images` drops the images.

The New Atlantis has its own extractor (`NewAtlantisExtractor`), which parses the page directly and needs no API key. Every other site goes through `LLMExtractor`, which sends the page to a model through Simon Willison's `llm`.

## Who depends on it

- **Brian's notes**: the YAML header goes straight into Markdown notes.
- **kokoro-reader's `recording-article-audio` skill** runs `content URL --no-images` for the text to narrate and `metadata URL` for the MP3 tags (title, artist, album) and the Audiobookshelf podcast name (`publication`).
- **kokoro-reader's narration server** (merged in kokoro-reader #10) publishes the metadata's `description` verbatim as the first line of each episode's public notes.
- **kokoro-reader's on-demand and nightly runner** (planned) reads `metadata --json` for the episode's title, authors, date, notes and edition. A key it relies on changing name, type or meaning breaks it without an error.

## What counts as an interface

- The subcommands, their flags, and the exit codes: 0 success; 1 when the fetch fails, `llm` is missing, or `content` can't extract the article; 2 for bad arguments (argparse). A failed LLM call in `metadata` is not an error: it prints a warning to stderr, returns the page's `<title>` (or "Unknown Title") with no authors, and exits 0.
- The YAML keys and their order, and the `periodical-edition` format `No. N (Season YYYY)`.
- The `--json` keys and types: `title` (string), `authors` (list of strings), `date_published` (string), `description`, `byline` (list of strings: the names shown under the title; `[]` when none is shown), `issue_number` (string), `issue_season`, `publication`. For The New Atlantis a missing key means the page had no value; a missing `byline` means the page had no title header (the `h1`'s parent `div.text-center`), so its byline is unknown; a page with the header and no name under the title has `[]`. kokoro-reader's runner reads `byline` aloud before the article, falling back to `authors` when it is missing. The LLM path returns only `title`, `authors`, `publication` and `date_published`, which can be empty (`""`, `[]`, `null`) and is left out when the call fails; it never returns `description` or an edition.
- Output on stdout only; diagnostics go to stderr. A stray `print` to stdout corrupts the JSON and the YAML for every caller.

## What a change can break

- **A wrong edition on a published episode.** The New Atlantis's site-wide nav links the current issue on every page, and "Related" cards link other issues. A search over the whole page attaches the wrong edition to an article; the article's own edition is the `/issues/` link in the header aside beside the `h1`.
- **A wrong description in public episode notes.** Which source the description comes from (meta description, `og:description`, the JSON-LD `WebPage` node) decides what listeners read; one live page sets `og:description` to just its byline.
- **A wrong or missing title, author or date** when the page's structured data changes shape. The site serves a Yoast JSON-LD `@graph` today, with authors as `@id` references and HTML entities left inside the strings.
- **A silent fallback.** When the structured data isn't found, the extractor falls back to the `h1` and the byline with no warning. A green suite can hide that the live site no longer matches the fixtures.
- **A failed LLM call that looks like success.** On generic sites, an expired key or a model error still exits 0, with placeholder metadata. The tests mock `llm`, so a change that makes every call fail stays green; check a generic URL live after touching `LLMExtractor`.

## What reviewers here tend to miss

- **The tests mock the network.** Every fixture is a hand-trimmed page. Run `metadata --json` against a few live pages before calling a parser change ready: a print-edition article, a web-only piece (no edition), an article from a double issue (such as "Losing Liberty in an Age of Access", No. 43, Summer/Fall 2014) and a multi-author piece. Run it on article pages: an `/issues/` page isn't an article and gives nonsense.
- **Web-only pieces and older layouts take different paths.** A web-only page has the header aside but no issue link; an older page has no header aside at all and still uses the page-wide search.
- **The YAML header changes when extraction changes.** A fix aimed at `--json` can change what lands in Brian's notes; check both outputs.
- **The repo's `CLAUDE.md` says the user manages commits.** Brian's rules for the bot account (his user-level `CLAUDE.md`) say to deliver work as a branch push plus a pull request, so a bot PR carries commits and says so in its description. Don't flag that as a violation.
