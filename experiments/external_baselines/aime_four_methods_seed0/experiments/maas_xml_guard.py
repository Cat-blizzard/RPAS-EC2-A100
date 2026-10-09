"""Bounded recovery for malformed MaAS ScEnsemble XML only.

This is an explicit *compatibility adapter*, not a substitute for native MaAS
controller search. Never fabricate a missing solution choice. All retry calls
still go through the native LLM bridge and are included in token telemetry.
"""
from __future__ import annotations

import re
from typing import Any


def install_scensemble_xml_guard(action_node_cls: type, usage: dict[str, Any], *, max_attempts: int = 2) -> None:
    """Install for ScEnsemble's sole `solution_letter` schema; leave other XML operations native."""
    if not (1 <= max_attempts <= 3):
        raise ValueError("max_attempts must be in [1,3] and be fixed before formal testing")
    original = action_node_cls.xml_fill
    if getattr(original, "_rpas_scensemble_xml_guard", False):
        raise RuntimeError("ScEnsemble XML adapter must only be installed once per process")

    usage.setdefault("maas_xml_invalid_responses", 0)
    usage.setdefault("maas_xml_recovered_samples", 0)
    usage.setdefault("maas_xml_exhausted_samples", 0)
    usage["maas_xml_max_attempts"] = max_attempts

    async def guarded(self: Any, context: str, images: Any = None) -> dict[str, Any]:
        fields = tuple(self.get_field_names())
        if fields != ("solution_letter",):
            return await original(self, context, images=images)
        for attempt in range(max_attempts):
            if attempt == 0:
                prompt = context
            else:
                # The role, answer mapping and candidate text remain unchanged.
                # Only make the native required XML response schema explicit.
                prompt = context + (
                    "\nYour previous output did not supply a valid solution_letter XML value."
                    " Return exactly one XML element such as"
                    " <solution_letter>A</solution_letter>, choosing the letter of the"
                    " correct existing candidate. No prose outside the XML element."
                )
            response = await original(self, prompt, images=images)
            letter = response.get("solution_letter", "") if isinstance(response, dict) else ""
            if isinstance(letter, str) and re.fullmatch(r"[A-Z]", letter.strip().upper()):
                if attempt:
                    usage["maas_xml_recovered_samples"] += 1
                return {**response, "solution_letter": letter.strip().upper()}
            usage["maas_xml_invalid_responses"] += 1
        usage["maas_xml_exhausted_samples"] += 1
        raise ValueError(
            f"MaAS ScEnsemble XML schema invalid after {max_attempts} attempts: "
            "missing or malformed solution_letter; no answer substituted"
        )

    guarded._rpas_scensemble_xml_guard = True
    action_node_cls.xml_fill = guarded
