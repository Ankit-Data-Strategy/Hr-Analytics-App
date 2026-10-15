"""
HR Attrition — Gemini API Integration
========================================
Sends the built prompt to Google's Gemini API and returns the consulting-
style narrative. Uses the official Google Gen AI SDK (`google-genai`).

Get a free API key at: https://aistudio.google.com/apikey
"""

import os
from google import genai


def generate_recommendations(prompt: str, api_key: str = None, model: str = "gemini-3.6-flash") -> str:
    """
    Call the Gemini API with the built prompt and return the response text.

    Parameters
    ----------
    prompt : str
        The full prompt from prompt_builder.build_recommendation_prompt().
    api_key : str, optional
        Google AI Studio / Gemini API key. If not provided, falls back to the
        GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.
    model : str
        Model ID to use. Google deprecates model IDs periodically and their
        API error messages tell you exactly what to switch to — if this ever
        404s again, check https://ai.google.dev/gemini-api/docs/models for
        the current recommended model ID and update this default.

    Returns
    -------
    str
        Gemini's response text.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key found. Pass api_key= directly, or set the "
            "GOOGLE_API_KEY environment variable."
        )

    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text
