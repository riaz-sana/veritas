"""Main verify() function — public entry point for Veritas."""
from __future__ import annotations
from veritas.core.config import Config, VeritasConfigError
from veritas.core.result import Verdict, VerificationResult
from veritas.orchestration.runner import VerificationRunner
from veritas.providers.claude import ClaudeProvider
from veritas.providers.search import BraveSearchProvider, TavilySearchProvider


async def verify(
    claim: str,
    context: str | None = None,
    domain: str | None = None,
    references: list[str] | None = None,
    model: str | None = None,
    config: Config | None = None,
    source_confidence: float | None = None,
) -> VerificationResult:
    """Verify a claim using adversarial parallel verification.

    Args:
        claim: The claim or AI output to verify.
        context: Optional source documents or surrounding context.
        domain: Domain hint (technical, scientific, medical, legal, code, schema, general).
        references: Optional list of reference document paths.
        model: Optional model override.
        config: Optional full configuration object.
        source_confidence: Optional confidence from the source model. If provided and
            config.confidence_routing is True, claims above the threshold skip verification.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if config is None:
        config = Config()
    if model:
        config.model = model
    config.validate()

    # Enterprise: confidence-based routing
    if (
        config.confidence_routing
        and source_confidence is not None
        and source_confidence >= config.confidence_threshold
    ):
        return VerificationResult(
            verdict=Verdict.VERIFIED,
            confidence=source_confidence,
            summary=f"Skipped verification — source confidence ({source_confidence:.2f}) above threshold ({config.confidence_threshold:.2f}).",
            failure_modes=[],
            evidence=[],
            contested=False,
            challenge_round=None,
            metadata={"skipped": True, "reason": "confidence_routing", "source_confidence": source_confidence},
        )

    # Enterprise: cache lookup
    cache = None
    if config.cache_enabled:
        from veritas.core.cache import VerdictCache
        cache = VerdictCache(db_path=config.cache_path, ttl_seconds=config.cache_ttl_seconds)
        cached = cache.get(claim, context, domain)
        if cached is not None:
            return cached

    # Run verification
    llm_provider = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)
    if config.search_provider == "tavily":
        search_provider = TavilySearchProvider(api_key=config.search_api_key)
    else:
        search_provider = BraveSearchProvider(api_key=config.search_api_key)
    runner = VerificationRunner(llm_provider=llm_provider, search_provider=search_provider, config=config)
    result = await runner.run(claim=claim, context=context, domain=domain, references=references or [])

    # Enterprise: cache store
    if cache is not None:
        cache.put(claim, context, domain, result)

    return result
