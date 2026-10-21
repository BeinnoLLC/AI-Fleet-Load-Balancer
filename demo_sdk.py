"""Drive the real TypeSafe SDK against the Ollama-backed OpenJev server.

Proves drop-in compatibility: the official typesafe-sdk talks to
http://127.0.0.1:8080 exactly as it would to OpenJev, answered by local Ollama.
"""
import os

os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:8080"
os.environ["TYPESAFE_API_KEY"] = "local-no-auth"  # SDK requires a key; server ignores it

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

EMAIL = (
    "I was charged twice for the same order #4471 this month. "
    "I need this fixed today or I'm cancelling my subscription."
)

with TypeSafeClient() as client:
    print("models:", [m.name for m in client.models.list().models])

    result = client.system_one(
        state=EMAIL,
        questions={
            "billing": Noul(instructions="Does this message concern a billing or payment problem?"),
            "tone": Choice(
                instructions="What is the emotional tone of the message?",
                criteria={"calm": "neutral, no emotion", "frustrated": "annoyed but patient",
                          "angry": "hostile, demanding", "threatening": "explicit cancellation threat"},
            ),
            "urgency": Score(
                instructions="Rate how urgent this email is.",
                criteria=["no urgency", "low", "moderate", "high", "critical, needs same-day action"],
            ),
        },
    )

    print("\n--- typed answers ---")
    print("billing.noul        :", result.nouls["billing"].noul)
    print("tone.choice         :", result.choices["tone"].choice,
          "| conf", round(result.choices["tone"].confidence, 3),
          "| probs", {k: round(v, 3) for k, v in result.choices["tone"].probabilities.items()})
    print("urgency.score       :", round(result.scores["urgency"].score, 2),
          "| probs", {k: round(v, 3) for k, v in result.scores["urgency"].probabilities.items()})
    print("usage               :", result.usage)
