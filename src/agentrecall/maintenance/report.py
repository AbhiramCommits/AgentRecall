"""Render audit findings as machine-readable JSON and a readable Markdown table."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agentrecall.maintenance.jobs import AuditResult
from agentrecall.models import utcnow


def report_payload(result: AuditResult, generated_at: datetime | None = None) -> dict:
    """Build the JSON-serializable audit payload."""
    generated_at = generated_at or utcnow()
    return {
        "generated_at": generated_at.isoformat(),
        "scanned": result.scanned,
        "total_findings": len(result.findings),
        "counts": result.counts,
        "findings": [finding.model_dump(mode="json") for finding in result.findings],
    }


def render_markdown(result: AuditResult, generated_at: datetime | None = None) -> str:
    """Render a Markdown audit report with a count summary and findings table."""
    generated_at = generated_at or utcnow()
    lines = [
        "# AgentRecall Knowledge-Repository Audit",
        "",
        f"Generated: {generated_at.isoformat()}",
        f"Entries scanned: {result.scanned}",
        f"Total findings: {len(result.findings)}",
        "",
        "## Findings by type",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    if result.counts:
        for kind, count in sorted(result.counts.items()):
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines += [
        "",
        "## Findings",
        "",
        "| Severity | Type | Entry | Detail |",
        "| --- | --- | --- | --- |",
    ]
    if result.findings:
        for finding in result.findings:
            detail = finding.detail.replace("|", "\\|")
            entry = finding.entry_id or "-"
            lines.append(f"| {finding.severity} | {finding.kind} | {entry} | {detail} |")
    else:
        lines.append("| - | none | - | No findings. |")
    lines.append("")
    return "\n".join(lines)


def write_report(
    result: AuditResult,
    out_dir: str | Path = "results",
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Write ``audit_report.json`` and ``audit_report.md`` and return their paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or utcnow()
    json_path = directory / "audit_report.json"
    md_path = directory / "audit_report.md"
    json_path.write_text(
        json.dumps(report_payload(result, generated_at), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(result, generated_at), encoding="utf-8")
    return json_path, md_path
