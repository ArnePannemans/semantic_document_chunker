import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def call_gemini(
    system_prompt,
    user_prompt,
    model,
    temperature,
    max_output_tokens,
    thinking_budget,
) -> str:
    """Call to Gemini model with thread-safe client creation to allow for parallel processing"""  # noqa: E501
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
        ),
    )
    return response.text
