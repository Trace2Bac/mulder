"""Evidence-derived strings must not become script in the HTML report.

Three ways a string read off the evidence could still execute after the
markup escaping in ``_markdown_to_safe_html`` and Jinja autoescaping:

* a markdown link keeps its target verbatim, so ``[x](javascript:...)`` in a
  finding description or the narrative became a live ``<a href>``;
* a source name or finding ``sources`` entry was placed inside a quoted JS
  string in an inline ``onclick``. Autoescaping turns ``'`` into ``&#39;``,
  but the browser decodes that before running the handler, so the quote
  still closes the string;
* the evidence browser tree, the evidence drawer and the command palette
  concatenated source names, finding titles and IOC values raw into
  ``innerHTML``.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

import pytest

from mulder.models import AuditSummary, CaseMetadataRow, Finding, SourceRow
from mulder.report.renderer import ReportRenderer, _markdown_to_safe_html

_SAFE_SCHEMES = {"http", "https", "mailto"}


class _Collector(HTMLParser):
    """Collect every URL attribute, every inline handler and every script body."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.handlers: list[str] = []
        self.evidence_sources: list[str] = []
        self.scripts: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name in ("href", "src"):
                self.urls.append(value)
            elif name.startswith("on"):
                self.handlers.append(value)
            elif name == "data-evidence-source":
                self.evidence_sources.append(value)
        self._in_script = tag == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.scripts.append(data)


def _parse(markup: str) -> _Collector:
    c = _Collector()
    c.feed(markup)
    return c


def _scheme(url: str) -> str | None:
    """The scheme a browser would see: entities decoded, controls and spaces dropped."""
    cleaned = re.sub(r"[\x00-\x20]", "", url)
    m = re.match(r"([A-Za-z][A-Za-z0-9+.\-]*):", cleaned)
    return m.group(1).lower() if m else None


def _assert_no_script_urls(urls: list[str]) -> None:
    for url in urls:
        scheme = _scheme(url)
        assert scheme is None or scheme in _SAFE_SCHEMES, url


# ---------------------------------------------------------------------------
# Markdown links
# ---------------------------------------------------------------------------


class TestMarkdownLinkSchemes:
    @pytest.mark.parametrize(
        "payload",
        [
            "[x](javascript:alert(1))",
            "[x](<javascript:alert(1)>)",
            "[x](JaVaScRiPt:alert(1))",
            "[x]( javascript:alert(1))",
            "[x](&#106;avascript:alert(1))",
            "[x](vbscript:msgbox(1))",
            "[x](data:text/html,<script>alert(1)</script>)",
            "[x][r]\n\n[r]: javascript:alert(1)",
            "![i](javascript:alert(1))",
        ],
    )
    def test_script_scheme_is_dropped(self, payload: str) -> None:
        out = _markdown_to_safe_html(f"See {payload}")
        _assert_no_script_urls(_parse(out).urls)

    def test_link_text_survives_when_href_is_dropped(self) -> None:
        out = _markdown_to_safe_html("[Open C2 panel](javascript:alert(1))")
        assert "Open C2 panel" in out

    @pytest.mark.parametrize(
        ("source", "href"),
        [
            ("[a](http://example.com)", "http://example.com"),
            ("[a](https://example.com/x?y=1)", "https://example.com/x?y=1"),
            ("[a](mailto:soc@example.com)", "mailto:soc@example.com"),
            ("[a](#finding-1)", "#finding-1"),
            ("[a](/relative/path)", "/relative/path"),
        ],
    )
    def test_safe_links_are_kept(self, source: str, href: str) -> None:
        assert href in _parse(_markdown_to_safe_html(source)).urls


# ---------------------------------------------------------------------------
# The rendered report
# ---------------------------------------------------------------------------

_TITLE = "Updater<img src=x onerror=alert(1)>"
_SOURCE_REF = "evtx.security');alert(1)//"
_SOURCE_NAME = "registry.ntuser.a')-alert(1)-('b"
_WINDOW_NAME = "pcap.disk.x<img src=x onerror=alert(1)>"
_PATH_IOC = "/etc/cron.d/<img/src=x/onerror=alert(1)>"


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp = tmp_path_factory.mktemp("xss")
    audit = tmp / "audit.jsonl"
    audit.write_text(
        '{"tool_call_id": "tc_1", "tool_name": "search", "params": {}, '
        '"duration_ms": 10, "timestamp": "2025-01-01T00:00:00Z"}\n'
    )
    meta = CaseMetadataRow(
        case_id="xss",
        ingested_at="2025-01-01T00:00:00Z",
        evidence_root="/evidence",
        extractor_versions={},
        narrative="Pivot via [the panel](javascript:alert(1)).",
    )
    finding = Finding(
        finding_id="f-1",
        case_id="xss",
        title=_TITLE,
        description=f"Planted link [Open C2 panel](JaVaScRiPt:alert(1)). Dropped {_PATH_IOC}",
        severity="critical",
        confidence="confirmed",
        evidence_refs=["tc_1"],
        sources=[_SOURCE_REF],
        submitted_at="2025-01-01T00:00:00Z",
    )
    summary = AuditSummary(
        total_tool_calls=1,
        total_findings=1,
        tool_call_counts={"search": 1},
        total_duration_ms=10,
        first_timestamp="2025-01-01T00:00:00Z",
        last_timestamp="2025-01-01T00:00:01Z",
    )
    sources = [
        SourceRow(
            source_id=1,
            case_id="xss",
            source_name=name,
            source_path="/evidence/x",
            source_hash="ab" * 32,
            extractor="test",
            line_count=1,
        )
        for name in (_SOURCE_NAME, _WINDOW_NAME)
    ]
    windows = {
        _SOURCE_NAME: [{"raw_text": "line"}],
        _WINDOW_NAME: [{"raw_text": "line"}],
    }
    return ReportRenderer().render_html(
        meta, [finding], summary, audit, sources_list=sources, source_windows=windows
    )


def test_no_script_url_in_report(report: str) -> None:
    _assert_no_script_urls(_parse(report).urls)


def test_no_evidence_string_reaches_an_inline_handler(report: str) -> None:
    """Handlers are decoded before they run, so escaping inside them is no defence."""
    for handler in _parse(report).handlers:
        assert "alert(1)" not in handler, handler


def test_hostile_values_stay_data_attributes(report: str) -> None:
    """The click targets now travel as data-* attributes and are still present."""
    targets = _parse(report).evidence_sources
    assert _SOURCE_NAME in targets
    assert _SOURCE_REF in targets


def _script(report: str) -> str:
    return "\n".join(_parse(report).scripts)


def test_script_does_not_build_inline_handlers_from_strings(report: str) -> None:
    """A JS-built ``onclick="f('...'+x+'...')"`` has the same decode-then-run hole."""
    assert "\\''+" not in _script(report)


@pytest.mark.parametrize("function", ["openDrawer", "buildEvSourceTree", "cmdSearch"])
def test_innerhtml_builders_escape_what_they_interpolate(report: str, function: str) -> None:
    """Every non-numeric value concatenated into these HTML strings goes through escHtml."""
    script = _script(report)
    start = script.index(f"function {function}(")
    body = script[start : script.index("\nfunction ", start + 1)]
    # Numbers, a class string built from constants, and syntaxHighlight (which
    # escapes before it adds its own spans).
    trusted = {"i", "cls", "rf.length", "groups[g].length", "(meta.line_count||'')"}
    trusted_prefixes = ("escHtml(", "syntaxHighlight(", "(entry.duration_ms", "Math.round(")
    for m in re.finditer(r"'\s*\+\s*(.+?)\s*\+\s*'", body):
        expr = m.group(1)
        if expr in trusted or expr.startswith(trusted_prefixes):
            continue
        pytest.fail(f"{function}: unescaped {expr!r}")


def test_report_still_contains_the_hostile_values_as_text(report: str) -> None:
    """Escaping must not lose the evidence the analyst needs to see."""
    assert html.escape(_TITLE, quote=False) in report
    assert "Open C2 panel" in report
    # The hostile path was extracted as an IOC and reaches the IOC table and palette.
    assert html.escape(_PATH_IOC, quote=False) in report
    assert "cron.d/\\u003cimg" in _script(report)
