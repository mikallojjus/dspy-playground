"""Global DSPy configuration for all models."""

import dspy
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Configure DSPy globally at module import (main thread, safe)
# This happens once when the module is first imported
_global_lm = None

try:
    _global_lm = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        num_ctx=32768,
    )
    dspy.configure(lm=_global_lm)
    logger.info(f"DSPy configured globally with {settings.ollama_model} at {settings.ollama_url}")
except Exception as e:
    logger.error(f"Failed to configure DSPy: {e}")
    raise


def get_lm_with_schema(schema: dict) -> dspy.LM:
    """
    Create an LM with JSON schema for guided decoding.

    Used for models that need structured output (e.g., ClaimExtractor).
    Models should use this with dspy.context() for schema-specific configuration.
    """
    return dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        format=schema,
        num_ctx=32768,
    )


def shutdown_dspy_configuration() -> None:
    """
    Shutdown DSPy configuration and close connections.

    This should be called during FastAPI shutdown to properly close
    HTTP connections to Ollama and release resources.
    """
    global _global_lm

    try:
        if _global_lm is not None:
            # Close any HTTP sessions/connections
            if hasattr(_global_lm, 'client') and _global_lm.client is not None:
                if hasattr(_global_lm.client, 'close'):
                    _global_lm.client.close()
                    logger.debug("Closed DSPy LM HTTP client")

            _global_lm = None
            logger.info("DSPy configuration shutdown complete")
    except Exception as e:
        logger.error(f"Error during DSPy shutdown: {e}", exc_info=True)
