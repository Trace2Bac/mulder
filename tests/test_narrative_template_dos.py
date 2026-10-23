"""The case narrative must not be evaluated as a Jinja2 template.

The narrative is model-authored prose summarising attacker-controlled
evidence. The renderer only needs to inject a handful of authoritative
counts (``{{finding_count}}``, ``{{mitre_techniques | length}}``, ...); any
other template construct -- loops, arithmetic, string multiplication -- is
not bounded by the Jinja2 sandbox, so nested ``range`` loops or
``'x' * 10**9`` wedge report generation. The payloads below are small
stand-ins: evaluating them at all is the bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from mulder.report.renderer import ReportRenderer

CTX: dict[str, Any] = {
    "finding_count": 7,
    "negative_count": 1,
    "confirmed_count": 4,
    "inference_count": 3,
    "critical_count": 2,
    "high_count": 3,
    "medium_count": 1,
    "low_count": 1,
    "sources_count": 5,
    "total_tool_calls": 120,
    "mitre_techniques": [{"id": "T1059"}, {"id": "T1021"}],
    "audit_log_path": "/srv/case/audit.jsonl",
    "findings": [{"title": "secret"}],
}

EVALUATED_PAYLOADS = [
    "{% for a in range(3) %}{% for b in range(3) %}x{% endfor %}{% endfor %}",
    "{{ 'x' * 5 }}",
    "{{ 2 ** 10 }}",
    "{{ finding_count * 1000 }}",
]


class TestNarrativeIsNotATemplate:
    @pytest.mark.parametrize("payload", EVALUATED_PAYLOADS)
    def test_template_constructs_are_not_evaluated(self, payload: str) -> None:
        assert ReportRenderer._render_narrative_template(payload, CTX) == payload

    def test_supported_placeholders_are_substituted(self) -> None:
        text = (
            "{{finding_count}} findings ({{ critical_count }} critical, "
            "{{high_count}} high, {{medium_count}} medium), "
            "{{confirmed_count}} confirmed, {{inference_count}} inferred, "
            "{{negative_count}} ruled out across {{sources_count}} sources and "
            "{{total_tool_calls}} tool calls; {{mitre_techniques | length}} techniques."
        )
        assert ReportRenderer._render_narrative_template(text, CTX) == (
            "7 findings (2 critical, 3 high, 1 medium), 4 confirmed, 3 inferred, "
            "1 ruled out across 5 sources and 120 tool calls; 2 techniques."
        )

    def test_non_numeric_context_is_not_exposed(self) -> None:
        text = "{{ audit_log_path }} {{ findings }}"
        assert ReportRenderer._render_narrative_template(text, CTX) == text

    def test_unknown_placeholder_is_left_verbatim(self) -> None:
        text = "{{ nonexistent_count }} and {% if x %}y{% endif %}"
        assert ReportRenderer._render_narrative_template(text, CTX) == text
