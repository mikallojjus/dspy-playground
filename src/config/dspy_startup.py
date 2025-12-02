"""DSPy startup validation for fail-fast behavior."""

import requests
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def validate_dspy_configuration() -> None:
    """
    Validate DSPy can connect to Ollama.

    This runs at FastAPI startup to ensure fail-fast behavior.
    Does NOT initialize heavy models, just validates connectivity.
    """
    try:
        # Test Ollama connectivity
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        response.raise_for_status()

        available_models = response.json().get("models", [])
        model_names = [m["name"] for m in available_models]

        if settings.ollama_model not in model_names:
            logger.warning(
                f"Model {settings.ollama_model} not found in Ollama. "
                f"Available: {model_names}"
            )

        logger.info(f"DSPy validation successful - Ollama reachable at {settings.ollama_url}")

    except requests.RequestException as e:
        logger.error(f"Failed to connect to Ollama at {settings.ollama_url}: {e}")
        raise RuntimeError(
            f"DSPy initialization failed: Cannot connect to Ollama at {settings.ollama_url}. "
            f"Ensure Ollama is running."
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during DSPy validation: {e}")
        raise
