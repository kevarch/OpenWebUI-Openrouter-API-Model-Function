"""
title: OpenRouter
version: 0.1.2
license: MIT
description: Adds support for OpenRouter, including citations and reasoning tokens
author: rburmorrison
author_url: https://github.com/rburmorrison
"""

import re
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field


def _insert_citations(text: str, citations: list[str]) -> str:
    """
    Replace citation markers [n] in text with markdown links to the corresponding citation URLs.

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in the text

    Returns:
        Text with citation markers replaced with markdown links
    """
    # Define regex pattern for citation markers [n]
    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        # Extract the number from the match
        num = int(match_obj.group(1))

        # Check if there's a corresponding citation URL
        # Citations are 0-indexed in the list, but 1-indexed in the text
        if 1 <= num <= len(citations):
            url = citations[num - 1]
            # Return Markdown link: [url]([n])
            return f"[{match_obj.group(0)}]({url})"
        else:
            # If no corresponding citation, return the original marker
            return match_obj.group(0)

    # Replace all citation markers in the text
    result = re.sub(pattern, replace_citation, text)

    return result


class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="", description="Your OpenRouter API key"
        )
        INCLUDE_REASONING: bool = Field(
            default=True,
            description="Request reasoning tokens from models that support it",
        )
        MODEL_PREFIX: str = Field(
            default="", description="Optional prefix for model names in Open WebUI"
        )

    def __init__(self):
        self.type = "manifold"  # Multiple models
        self.valves = self.Valves()

    def pipes(self) -> List[dict]:
        """Fetch available models from OpenRouter API"""
        if not self.valves.OPENROUTER_API_KEY:
            return [{"id": "error", "name": "API Key not provided"}]

        try:
            headers = {"Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}"}
            response = requests.get(
                "https://openrouter.ai/api/v1/models", headers=headers
            )

            if response.status_code != 200:
                return [
                    {
                        "id": "error",
                        "name": f"Error fetching models: {response.status_code}",
                    }
                ]

            models_data = response.json()

            # Extract model information
            models = []
            for model in models_data.get("data", []):
                model_id = model.get("id")
                if model_id:
                    # Use model name or ID, with optional prefix
                    model_name = model.get("name", model_id)
                    prefix = self.valves.MODEL_PREFIX
                    models.append(
                        {
                            "id": model_id,
                            "name": f"{prefix}{model_name}" if prefix else model_name,
                        }
                    )

            return models or [{"id": "error", "name": "No models found"}]

        except Exception as e:
            print(f"Error fetching models: {e}")
            return [{"id": "error", "name": f"Error: {str(e)}"}]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """Process the request and handle reasoning tokens if supported"""
        # Clone the body for OpenRouter
        payload = body.copy()

        # Print incoming body for debugging
        print(f"Original request body: {json.dumps(body)[:500]}...")

        # Make sure the model ID is properly extracted from the pipe format
        if "model" in payload and payload["model"] and "." in payload["model"]:
            # Extract the model ID from the format like "openrouter.model-id"
            payload["model"] = payload["model"].split(".", 1)[1]
            print(f"Extracted model ID: {payload['model']}")

        # Add include_reasoning parameter if enabled
        if self.valves.INCLUDE_REASONING:
            payload["include_reasoning"] = True

        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        # Add HTTP-Referer and X-Title if provided
        # These help identify your app on OpenRouter
        if body.get("http_referer"):
            headers["HTTP-Referer"] = body["http_referer"]
        if body.get("x_title"):
            headers["X-Title"] = body["x_title"]

        # Default headers for identifying the app to OpenRouter
        if "HTTP-Referer" not in headers:
            headers["HTTP-Referer"] = "https://openwebui.com/"
        if "X-Title" not in headers:
            headers["X-Title"] = "Open WebUI via Pipe"

        url = "https://openrouter.ai/api/v1/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        """Handle non-streaming responses and wrap reasoning in <think> tags if present"""
        try:
            print(
                f"Sending non-streaming request to OpenRouter: {json.dumps(payload)[:200]}..."
            )
            response = requests.post(url, headers=headers, json=payload, timeout=90)

            if response.status_code != 200:
                error_message = f"HTTP Error {response.status_code}"
                try:
                    error_data = response.json()
                    print(f"Error response: {json.dumps(error_data)}")
                    if "error" in error_data:
                        if (
                            isinstance(error_data["error"], dict)
                            and "message" in error_data["error"]
                        ):
                            error_message += f": {error_data['error']['message']}"
                        else:
                            error_message += f": {error_data['error']}"
                except Exception as e:
                    print(f"Failed to parse error response: {e}")
                    error_message += f": {response.text[:500]}"

                # Log request payload for debugging
                print(f"Request that caused error: {json.dumps(payload)}")
                raise Exception(error_message)

            res = response.json()
            print(f"OpenRouter response keys: {list(res.keys())}")

            # Check if we have choices in the response
            if not res.get("choices") or len(res["choices"]) == 0:
                return ""

            # Extract content and reasoning if present
            choice = res["choices"][0]
            message = choice.get("message", {})

            # Debug output
            print(f"Message keys: {list(message.keys())}")

            content = message.get("content", "")
            reasoning = message.get("reasoning", "")

            print(f"Found reasoning: {bool(reasoning)} ({len(reasoning)} chars)")
            print(f"Found content: {bool(content)} ({len(content)} chars)")

            # If we have both reasoning and content
            if reasoning and content:
                return f"<think>\n{reasoning}\n</think>\n\n{content}"
            elif reasoning:  # Only reasoning, no content (unusual)
                return f"<think>\n{reasoning}\n</think>\n\n"
            elif content:  # Only content, no reasoning
                return content
            return ""
        except Exception as e:
            print(f"Error in non_stream_response: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        """Stream reasoning tokens in real-time with proper tag management"""
        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=True, timeout=90
            )

            if response.status_code != 200:
                error_message = f"HTTP Error {response.status_code}"
                try:
                    error_data = response.json()
                    error_message += (
                        f": {error_data.get('error', {}).get('message', '')}"
                    )
                except:
                    pass
                raise Exception(error_message)

            # State tracking
            in_reasoning_state = False  # True if we've output the opening <think> tag
            latest_citations = []  # The latest citations list

            # Process the response stream
            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "):
                    continue
                elif line_text == "data: [DONE]":
                    if latest_citations:
                        citation_list = [f"1. {l}" for l in latest_citations]
                        citation_list_str = "\n".join(citation_list)
                        yield f"\n\n---\nCitations:\n{citation_list_str}"
                    continue

                try:
                    chunk = json.loads(line_text[6:])

                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        citations = chunk.get("citations") or []

                        # Update the citation list
                        if citations:
                            latest_citations = citations

                        # Check for reasoning tokens
                        reasoning_text = None
                        if "delta" in choice and "reasoning" in choice["delta"]:
                            reasoning_text = choice["delta"]["reasoning"]
                        elif "message" in choice and "reasoning" in choice["message"]:
                            reasoning_text = choice["message"]["reasoning"]

                        # Check for content tokens
                        content_text = None
                        if "delta" in choice and "content" in choice["delta"]:
                            content_text = choice["delta"]["content"]
                        elif "message" in choice and "content" in choice["message"]:
                            content_text = choice["message"]["content"]

                        # Handle reasoning tokens
                        if reasoning_text:
                            # If first reasoning token, output opening tag
                            if not in_reasoning_state:
                                yield "<think>\n"
                                in_reasoning_state = True

                            # Output the reasoning token
                            yield _insert_citations(reasoning_text, citations)

                        # Handle content tokens
                        if content_text:
                            # If transitioning from reasoning to content, close the thinking tag
                            if in_reasoning_state:
                                yield "\n</think>\n\n"
                                in_reasoning_state = False

                            # Output the content
                            if content_text:
                                yield _insert_citations(content_text, citations)

                except Exception as e:
                    print(f"Error processing chunk: {e}")

            # If we're still in reasoning state at the end, close the tag
            if in_reasoning_state:
                yield "\n</think>\n\n"

        except Exception as e:
            print(f"Error in stream_response: {e}")
            yield f"Error: {e}"
