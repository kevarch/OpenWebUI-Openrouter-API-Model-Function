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


# --- Helper function (unchanged) ---
def _insert_citations(text: str, citations: list[str]) -> str:
    """
    Replace citation markers [n] in text with markdown links to the corresponding citation URLs.

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in the text

    Returns:
        Text with citation markers replaced with markdown links
    """
    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        num = int(match_obj.group(1))
        if 1 <= num <= len(citations):
            url = citations[num - 1]
            return f"[{match_obj.group(0)}]({url})"
        else:
            return match_obj.group(0)

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

            models = []
            for model in models_data.get("data", []):
                model_id = model.get("id")
                if model_id:
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
        payload = body.copy()
        print(f"Original request body: {json.dumps(body)[:500]}...")

        if "model" in payload and payload["model"] and "." in payload["model"]:
            payload["model"] = payload["model"].split(".", 1)[1]
            print(f"Extracted model ID: {payload['model']}")

        if self.valves.INCLUDE_REASONING:
            payload["include_reasoning"] = True

        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        if body.get("http_referer"):
            headers["HTTP-Referer"] = body["http_referer"]
        if body.get("x_title"):
            headers["X-Title"] = body["x_title"]

        if "HTTP-Referer" not in headers:
            headers["HTTP-Referer"] = "https://openwebui.com/"
        if "X-Title" not in headers:
            headers["X-Title"] = "Open WebUI via Pipe"

        url = "https://openrouter.ai/api/v1/chat/completions"

        try:
            if body.get("stream", False):
                # Pass the helper function as an argument
                return self.stream_response(url, headers, payload, _insert_citations)
            else:
                # Pass the helper function as an argument
                return self.non_stream_response(url, headers, payload, _insert_citations)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def non_stream_response(self, url, headers, payload, citation_inserter):
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

                print(f"Request that caused error: {json.dumps(payload)}")
                raise Exception(error_message)

            res = response.json()
            print(f"OpenRouter response keys: {list(res.keys())}")

            if not res.get("choices") or len(res["choices"]) == 0:
                return ""

            choice = res["choices"][0]
            message = choice.get("message", {})
            citations = res.get("citations") or [] # Get citations if present

            print(f"Message keys: {list(message.keys())}")

            content = message.get("content") # Get content, might be None
            reasoning = message.get("reasoning") # Get reasoning, might be None

            # Explicitly check for None and default to empty string
            if content is None:
                content = ""
            if reasoning is None:
                reasoning = ""

            print(f"Found reasoning: {bool(reasoning)} ({len(reasoning)} chars)")
            print(f"Found content: {bool(content)} ({len(content)} chars)")

            # Insert citations into content and reasoning
            content_with_citations = citation_inserter(content, citations)
            reasoning_with_citations = citation_inserter(reasoning, citations)

            # Combine reasoning and content
            if reasoning_with_citations and content_with_citations:
                # Add citation list at the end if citations exist
                citation_list_str = ""
                if citations:
                    citation_list = [f"{i+1}. {url}" for i, url in enumerate(citations)]
                    citation_list_str = "\n\n---\nCitations:\n" + "\n".join(citation_list)

                return f"<think>\n{reasoning_with_citations}\n</think>\n\n{content_with_citations}{citation_list_str}"
            elif reasoning_with_citations:
                return f"<think>\n{reasoning_with_citations}\n</think>\n\n"
            elif content_with_citations:
                 # Add citation list at the end if citations exist
                citation_list_str = ""
                if citations:
                    citation_list = [f"{i+1}. {url}" for i, url in enumerate(citations)]
                    citation_list_str = "\n\n---\nCitations:\n" + "\n".join(citation_list)
                return f"{content_with_citations}{citation_list_str}"

            return "" # Return empty string if neither content nor reasoning exists
        except Exception as e:
            # Print traceback for better debugging
            import traceback
            traceback.print_exc()
            print(f"Error in non_stream_response: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload, citation_inserter):
        """Stream reasoning tokens in real-time with proper tag management"""
        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=True, timeout=90
            )

            if response.status_code != 200:
                error_message = f"HTTP Error {response.status_code}"
                try:
                    # Try to read the error response body
                    error_text = response.text
                    print(f"Error response body: {error_text[:500]}") # Log the first 500 chars
                    error_data = json.loads(error_text) # Try parsing after reading
                    error_detail = error_data.get('error', {}).get('message', '')
                    if error_detail:
                         error_message += f": {error_detail}"
                    elif error_text:
                         error_message += f": {error_text[:200]}" # Add raw text if no structured error
                except json.JSONDecodeError:
                     error_message += f": {response.text[:200]}" # Add raw text if JSON parsing fails
                except Exception as e:
                    print(f"Failed to fully parse error response: {e}")
                    error_message += f": {response.text[:200]}" # Fallback

                print(f"Request that caused error: {json.dumps(payload)}")
                raise Exception(error_message)


            in_reasoning_state = False
            latest_citations = []
            buffer = "" # Buffer to accumulate text before processing citations

            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "):
                    continue
                elif line_text == "data: [DONE]":
                    # Process any remaining buffer content with the latest citations
                    if buffer:
                        yield citation_inserter(buffer, latest_citations)
                        buffer = "" # Clear buffer

                    # Add final citation list if needed
                    if latest_citations:
                        citation_list = [f"{i+1}. {url}" for i, url in enumerate(latest_citations)]
                        citation_list_str = "\n".join(citation_list)
                        yield f"\n\n---\nCitations:\n{citation_list_str}"
                    break # Exit the loop cleanly

                try:
                    chunk_data = line_text[6:]
                    # Handle potential empty data chunks if API sends "data: \n"
                    if not chunk_data.strip():
                        continue
                    chunk = json.loads(chunk_data)

                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        # Update citations if present in the chunk
                        chunk_citations = chunk.get("citations")
                        if chunk_citations is not None: # Check specifically for None
                             latest_citations = chunk_citations

                        delta = choice.get("delta", {})
                        message = choice.get("message", {}) # Handle non-delta messages too

                        reasoning_text = delta.get("reasoning") if delta else message.get("reasoning")
                        content_text = delta.get("content") if delta else message.get("content")

                        # --- FIX: Ensure None is treated as empty string ---
                        reasoning_text = reasoning_text if reasoning_text is not None else ""
                        content_text = content_text if content_text is not None else ""
                        # --- END FIX ---


                        # Handle reasoning tokens
                        if reasoning_text:
                            if not in_reasoning_state:
                                # Process buffer before starting <think> tag
                                if buffer:
                                    yield citation_inserter(buffer, latest_citations)
                                    buffer = ""
                                yield "<think>\n"
                                in_reasoning_state = True
                            # Append reasoning text to buffer
                            buffer += reasoning_text

                        # Handle content tokens
                        if content_text:
                            if in_reasoning_state:
                                # Process buffer (reasoning) before closing </think> tag
                                if buffer:
                                     yield citation_inserter(buffer, latest_citations)
                                     buffer = ""
                                yield "\n</think>\n\n"
                                in_reasoning_state = False
                            # Append content text to buffer
                            buffer += content_text

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line_text}")
                except Exception as e:
                    print(f"Error processing chunk: {e} - Line: {line_text}")
                    import traceback
                    traceback.print_exc()


            # If we end while still in reasoning state, close the tag and process buffer
            if in_reasoning_state:
                 if buffer:
                     yield citation_inserter(buffer, latest_citations) # Process remaining reasoning
                     buffer = ""
                 yield "\n</think>\n\n"
            # Process any remaining buffer content (could be content if stream ends abruptly)
            elif buffer:
                 yield citation_inserter(buffer, latest_citations)


        except Exception as e:
            print(f"Error in stream_response: {e}")
            import traceback
            traceback.print_exc()
            # Yield the error message to the client
            yield f"Error: An error occurred during streaming - {e}"
