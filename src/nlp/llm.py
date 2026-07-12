"""LLM provider abstraction: anthropic | local (flan-t5-base) | template.

``generate(system, user)`` has the same signature for all providers.
``template`` is the default and the defence-day safety net: deterministic
Jinja rendering, zero network, zero API keys. Falls back template <- local
<- anthropic on any provider error so the demo can never crash here.
"""

from __future__ import annotations

import logging

from config import settings

log = logging.getLogger(__name__)

_local_pipe = None


def _generate_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _generate_local(system: str, user: str) -> str:
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline

        _local_pipe = pipeline("text2text-generation", model="google/flan-t5-base",
                               device=-1)
    out = _local_pipe(f"{system}\n\n{user}", max_new_tokens=300, do_sample=False)
    return out[0]["generated_text"].strip()


def _generate_template(system: str, user: str) -> str:
    """Deterministic fallback: the caller (advisor/summarizer) is expected to
    pass a pre-rendered template text as ``user`` when provider=template.

    For template mode the "generation" is identity: advisor and summarizer
    build the full Azerbaijani text themselves via Jinja and route it through
    here so all three providers share one code path.
    """
    return user.strip()


def generate(system: str, user: str) -> str:
    """Generate text with the configured provider, falling back safely."""
    provider = settings.llm_provider
    if provider == "anthropic" and settings.anthropic_api_key:
        try:
            return _generate_anthropic(system, user)
        except Exception:  # noqa: BLE001 - any API failure must not kill the demo
            log.exception("anthropic provider failed, falling back to local")
            provider = "local"
    if provider == "local":
        try:
            return _generate_local(system, user)
        except Exception:  # noqa: BLE001
            log.exception("local provider failed, falling back to template")
    return _generate_template(system, user)


def active_provider() -> str:
    """The provider that would actually be used right now."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        return "anthropic"
    return settings.llm_provider


if __name__ == "__main__":
    print(f"provider={active_provider()}")
    print(generate("test system", "salam, bu template cavabıdır"))
