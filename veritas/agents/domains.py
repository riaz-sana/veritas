"""Domain-specific agent prompt extensions.

When domain is set, agents get additional instructions tailored to that domain.
These extend (not replace) the base agent prompts.

Supported domains:
- general (default — no extra instructions)
- technical
- scientific
- medical
- legal
- code (NEW — for verifying generated code against specs)
- schema (NEW — for verifying data models against SQL or vice versa)
"""

from __future__ import annotations

# Domain-specific additions appended to each agent's system prompt

LOGIC_DOMAINS: dict[str, str] = {
    "code": """
Additional rules for CODE verification:
- Check if the code implements what the spec describes
- Look for missing edge case handling
- Verify function signatures match the spec
- Check for logical errors in control flow (off-by-one, wrong conditions)
- Verify error handling matches the documented behavior
- Check if all spec requirements are addressed in the code
""",
    "schema": """
Additional rules for SCHEMA verification:
- Check if all entities in the source are represented in the target
- Verify foreign key relationships are correct and complete
- Check that column types match the data being stored
- Verify NOT NULL constraints are appropriate
- Check for missing indexes on frequently queried columns
- Verify that the schema normalization level is consistent
""",
    "medical": """
Additional rules for MEDICAL verification:
- Pay special attention to dosage numbers and units
- Check if drug interactions or contraindications are mentioned when relevant
- Verify that medical claims reference appropriate evidence levels
- Flag claims that present correlational evidence as causal
- Check temporal relevance of medical guidelines (guidelines change frequently)
""",
    "legal": """
Additional rules for LEGAL verification:
- Check jurisdiction specificity (laws vary by jurisdiction)
- Verify that legal citations are formatted correctly
- Flag broad legal claims that lack jurisdiction context
- Check if statutes of limitation or effective dates are relevant
- Verify that regulatory references are current
""",
    "scientific": """
Additional rules for SCIENTIFIC verification:
- Check if claims about study results match the described methodology
- Verify that statistical claims are internally consistent (sample sizes, p-values, effect sizes)
- Flag claims that conflate correlation with causation
- Check if the claim accurately represents the study's scope and limitations
""",
    "financial": """
Additional rules for FINANCIAL verification:
- Verify numerical accuracy (revenue, percentages, growth rates)
- Check if financial metrics are internally consistent (e.g., revenue - expenses = profit)
- Verify time periods are correctly stated
- Flag forward-looking statements presented as facts
""",
}

SOURCE_DOMAINS: dict[str, str] = {
    "code": """
Additional rules for CODE source verification:
- Compare the implementation against the spec line by line
- Check if all functions/classes mentioned in the spec exist in the code
- Verify return types match what the spec describes
- Check if the code handles all the input/output cases in the spec
- Look for hardcoded values that should be configurable per spec
""",
    "schema": """
Additional rules for SCHEMA source verification:
- Verify every table/column in the source model appears in the generated SQL
- Check that data types are appropriate (VARCHAR lengths, INT vs BIGINT, etc.)
- Verify join conditions match the entity relationships
- Check if the source mentions constraints that are missing in the SQL
- Verify index definitions match query patterns described in the source
""",
    "medical": """
Additional rules for MEDICAL source verification:
- Cross-reference drug names, dosages, and administration routes
- Verify ICD codes or medical terminology accuracy
- Check if cited studies exist and report what is claimed
""",
    "legal": """
Additional rules for LEGAL source verification:
- Verify statute numbers and section references
- Cross-reference case citations if present
- Check if regulatory references are to the correct agency
""",
}

ADVERSARY_DOMAINS: dict[str, str] = {
    "code": """
Additional adversarial strategies for CODE:
- Try to find inputs that would cause the code to crash or behave incorrectly
- Look for race conditions or concurrency issues
- Check for security vulnerabilities (injection, overflow, unauthorized access)
- Try edge cases: empty inputs, null values, maximum values, negative numbers
- Check if the code handles errors the same way the spec describes
""",
    "schema": """
Additional adversarial strategies for SCHEMA:
- Try to construct data that would violate the schema constraints
- Look for circular foreign key dependencies
- Check if deletion cascades could cause data loss
- Try to find queries the schema can't efficiently support
- Check for data that fits the schema but violates business rules
""",
    "medical": """
Additional adversarial strategies for MEDICAL:
- Check if the claim could be dangerous if wrong (dosage errors, drug interactions)
- Look for overgeneralization of treatment effectiveness across populations
- Check if side effects or risks are understated
""",
    "legal": """
Additional adversarial strategies for LEGAL:
- Check if the legal advice applies to all jurisdictions or only specific ones
- Look for exceptions or recent amendments that change the conclusion
- Check if the claim oversimplifies complex legal nuance
""",
}

CALIBRATION_DOMAINS: dict[str, str] = {
    "code": """
Additional calibration rules for CODE:
- Code that "works" is not the same as code that "matches the spec" — assess confidence in SPEC COMPLIANCE, not just correctness
- Simple utility functions warrant higher confidence than complex business logic
- If the spec is ambiguous, confidence should be lower regardless of code quality
""",
    "schema": """
Additional calibration rules for SCHEMA:
- Simple CRUD schemas warrant higher confidence than complex analytical schemas
- If the source model has ambiguous relationships, confidence should be lower
- Schema migrations have different confidence profiles than greenfield schemas
""",
    "medical": """
Additional calibration rules for MEDICAL:
- Medical claims should generally have LOWER confidence unless backed by meta-analyses
- Case studies and single trials warrant much lower confidence than systematic reviews
- "Promising" results should not be presented with high confidence
""",
    "legal": """
Additional calibration rules for LEGAL:
- Legal conclusions should have lower confidence when jurisdiction is unspecified
- Settled law warrants higher confidence than emerging legal doctrine
- Multi-factor legal tests should have lower confidence than bright-line rules
""",
}


def get_domain_extension(agent_type: str, domain: str | None) -> str:
    """Get domain-specific prompt extension for an agent.

    Args:
        agent_type: "logic", "source", "adversary", or "calibration"
        domain: The domain hint, or None for no extension

    Returns:
        Additional prompt text to append, or empty string
    """
    if not domain:
        return ""

    domain = domain.lower()
    extensions = {
        "logic": LOGIC_DOMAINS,
        "source": SOURCE_DOMAINS,
        "adversary": ADVERSARY_DOMAINS,
        "calibration": CALIBRATION_DOMAINS,
    }

    agent_extensions = extensions.get(agent_type, {})
    return agent_extensions.get(domain, "")
