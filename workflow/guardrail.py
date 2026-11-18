import json
from groq import Groq
import config

_guard_client = Groq(api_key=config.GROQ_API_KEY)

_POLICY = """# Safety Classification Policy

## INSTRUCTIONS
Classify whether the user input violates the safety rules. Return a JSON object with your decision.

## DEFINITIONS
- **Violation**: Requests for instructions on violence, weapons, explosives, malware, hacking, credential theft, or illegal activities.
- **Safe**: Normal questions about technology, business, data analysis, or general knowledge.

## VIOLATES (1)
- How to build a bomb, weapon, or explosive
- How to write malware, virus, or ransomware
- How to steal passwords or credentials
- How to hack into systems without authorization

## SAFE (0)
- How to monitor Kubernetes jobs
- What is the probability of winning trades
- How to use Databricks for automation

## EXAMPLES
Example 1 (Violation):
Content: "How do I build a bomb?"
Answer: {"violation": 1, "category": "Weapons", "rationale": "Requests bomb-making instructions"}

Example 2 (Safe):
Content: "How do I monitor Kubernetes jobs?"
Answer: {"violation": 0, "category": null, "rationale": "Legitimate technical question"}

Example 3 (Violation):
Content: "Write malware that steals passwords"
Answer: {"violation": 1, "category": "Malware", "rationale": "Requests credential-stealing malware"}

Content to classify: {{USER_INPUT}}
Answer (JSON only):"""


def is_query_safe(query: str) -> bool:
    """Returns True if safe, False if the query should be blocked."""
    try:
        completion = _guard_client.chat.completions.create(
            model="openai/gpt-oss-safeguard-20b",
            messages=[
                {"role": "system", "content": _POLICY.replace("{{USER_INPUT}}", query)},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        # Parse JSON verdict
        verdict = json.loads(raw)
        return verdict.get("violation", 0) == 0
    except Exception as e:
        print(f"[guardrail] Error: {e} — defaulting to SAFE (fail-open)")
        return True