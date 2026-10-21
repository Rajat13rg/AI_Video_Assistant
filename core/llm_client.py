
import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

# Keep requests paced within this running application.
openai_limiter = InMemoryRateLimiter(
    requests_per_second=0.75,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing from the environment."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        api_key=api_key,
        max_retries=5,
        timeout=120,
        rate_limiter=openai_limiter,
    )
