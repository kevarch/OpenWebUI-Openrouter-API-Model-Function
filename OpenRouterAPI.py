"""
title: OpenRouter Integration for OpenWebUI
version: 0.2.0
description: Access the full suite of OpenRouter models directly within OpenWebUI with support for citations and reasoning tokens.
author: kevarch
author_url: https://github.com/kevarch
credits: rburmorrison (https://github.com/rburmorrison), Google Gemini Pro 2.5
license: MIT
"""

import re
import requests
import json
import traceback # Import traceback for detailed error logging
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field


# --- Helper function for citation text insertion ---
def _insert_citations(text: str, citations: list[str]) -> str:
    """
    Replace citation markers [n] in text with markdown links to the corresponding citation URLs.

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in the text

    Returns:
        Text with citation markers replaced with markdown links
    """
    if not citations or not text: # No citations or text, return text as is
        return text

    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        try:
            num = int(match_obj.group(1))
            # Citations are 0-indexed in the list, but 1-indexed in the text
            if 1 <= num <= len(citations):
                url = citations[num - 1]
                # Return Markdown link: [[n]](url) - common format
                return f"[[{num}]]({url})"
            else:
                # If no corresponding citation, return the original marker
                return match_obj.group(0)
        except (ValueError, IndexError):
             # Handle cases where the number is invalid or index out of bounds
             return match_obj.group(0)

    try:
        result = re.sub(pattern, replace_citation, text)
        return result
    except Exception as e:
        print(f"Error during citation insertion: {e}")
        return text # Return original text if regex fails


# --- Helper function for formatting the final citation list ---
def _format_citation_list(citations: list[str]) -> str:
    """
    Formats a list of citation URLs into a markdown string.

    Args:
        citations: A list of citation URLs.

    Returns:
        A formatted markdown string (e.g., "\n\n---\nCitations:\n1. url1\n2. url2")
        or an empty string if no citations are provided.
    """
    if not citations:
        return ""

    try:
        citation_list = [f"{i+1}. {url}" for i, url in enumerate(citations)]
        return "\n\n---\nCitations:\n" + "\n".join(citation_list)
    except Exception as e:
        print(f"Error formatting citation list: {e}")
        return "" # Return empty string on error


# --- Main Pipe class ---
class Pipe:
    class Valves(BaseModel):
        # User-configurable settings
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your OpenRouter API key (required)."
        )
        INCLUDE_REASONING: bool = Field(
            default=True,
            description="Request reasoning tokens from models that support it.",
        )
        MODEL_PREFIX: str = Field(
            default="",
            description="Optional prefix for model names in Open WebUI (e.g., 'OR: ').",
        )
        # NEW: Configurable request timeout
        REQUEST_TIMEOUT: int = Field(
            default=90,
            description="Timeout for API requests in seconds.",
            gt=0 # Ensure timeout is positive
        )

    def __init__(self):
        self.type = "manifold"  # Specifies this pipe provides multiple models
        self.valves = self.Valves()
        # Simple check on init, though user might change valves later
        if not self.valves.OPENROUTER_API_KEY:
            print("Warning: OPENROUTER_API_KEY is not set in Valves.")


    def pipes(self) -> List[dict]:
        """
        Fetches available models from the OpenRouter API.
        This method is called by OpenWebUI to discover the models this pipe provides.
        """
        if not self.valves.OPENROUTER_API_KEY:
            # Return an error entry if API key is missing
            return [{"id": "error", "name": "Pipe Error: OpenRouter API Key not provided"}]

        try:
            headers = {"Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}"}
            # Use configured timeout for this request too
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=self.valves.REQUEST_TIMEOUT
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            models_data = response.json()
            models = []
            for model in models_data.get("data", []):
                model_id = model.get("id")
                if model_id:
                    model_name = model.get("name", model_id) # Use name, fallback to id
                    prefix = self.valves.MODEL_PREFIX
                    models.append(
                        {
                            "id": model_id, # The actual ID OpenRouter expects
                            # Display name in OpenWebUI, potentially prefixed
                            "name": f"{prefix}{model_name}" if prefix else model_name,
                        }
                    )

            if not models:
                 return [{"id": "error", "name": "Pipe Error: No models found on OpenRouter"}]

            return models

        # --- Improved Error Handling ---
        except requests.exceptions.Timeout:
            print("Error fetching models: Request timed out.")
            return [{"id": "error", "name": "Pipe Error: Timeout fetching models"}]
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (like 401 Unauthorized, 404 Not Found, etc.)
            error_msg = f"Pipe Error: HTTP {e.response.status_code} fetching models"
            try:
                 # Try to get more specific error from response body
                 error_detail = e.response.json().get("error", {}).get("message", "")
                 if error_detail:
                     error_msg += f": {error_detail}"
            except json.JSONDecodeError:
                 pass # Ignore if response body isn't valid JSON
            print(f"Error fetching models: {error_msg} (URL: {e.request.url})")
            return [{"id": "error", "name": error_msg}]
        except requests.exceptions.RequestException as e:
            # Handle other network/request errors
            print(f"Error fetching models: Request failed: {e}")
            return [{"id": "error", "name": f"Pipe Error: Network error fetching models: {e}"}]
        except Exception as e:
            # Catch any other unexpected errors during model fetching
            print(f"Unexpected error fetching models: {e}")
            traceback.print_exc()
            return [{"id": "error", "name": f"Pipe Error: Unexpected error: {e}"}]


    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """
        Processes incoming chat requests. This is the main function called by OpenWebUI
        when a user interacts with a model provided by this pipe.

        Args:
            body: The request body, conforming to OpenAI chat completions format.

        Returns:
            Either a string (for non-streaming responses) or a generator/iterator
            (for streaming responses).
        """
        if not self.valves.OPENROUTER_API_KEY:
             # Handle missing API key at the start of processing
             return "Pipe Error: OpenRouter API Key is not configured."

        try:
            payload = body.copy()
            # print(f"Original request body: {json.dumps(body)[:500]}...") # Optional: for debugging

            # Extract the actual model ID if it's prefixed (e.g., "pipe_id.model_id")
            # OpenWebUI might send the model field like this.
            if "model" in payload and payload["model"] and "." in payload["model"]:
                # Assuming format "some_prefix.actual-model-id"
                payload["model"] = payload["model"].split(".", 1)[1]
                # print(f"Extracted model ID: {payload['model']}") # Optional: for debugging

            # Add include_reasoning parameter if the valve is enabled
            if self.valves.INCLUDE_REASONING:
                payload["include_reasoning"] = True

            # Prepare headers for the OpenRouter API request
            headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                # Recommended headers for OpenRouter analytics
                "HTTP-Referer": body.get("http_referer", "https://openwebui.com/"), # Default if not provided
                "X-Title": body.get("x_title", "Open WebUI via Pipe"), # Default if not provided
            }

            # API endpoint for chat completions
            url = "https://openrouter.ai/api/v1/chat/completions"

            # Determine if streaming is requested
            is_streaming = body.get("stream", False)

            # Call the appropriate handler based on streaming mode
            if is_streaming:
                # Pass helper functions and timeout value
                return self.stream_response(
                    url, headers, payload, _insert_citations, _format_citation_list, self.valves.REQUEST_TIMEOUT
                )
            else:
                # Pass helper functions and timeout value
                return self.non_stream_response(
                    url, headers, payload, _insert_citations, _format_citation_list, self.valves.REQUEST_TIMEOUT
                )

        except Exception as e:
            # Catch unexpected errors during request preparation
            print(f"Error preparing request in pipe method: {e}")
            traceback.print_exc()
            # Return error string (will be displayed to user in UI)
            return f"Pipe Error: Failed to prepare request: {e}"


    def non_stream_response(self, url, headers, payload, citation_inserter, citation_formatter, timeout):
        """Handles non-streaming API requests."""
        try:
            # print(f"Sending non-streaming request to OpenRouter: {json.dumps(payload)[:200]}...") # Optional
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status() # Check for HTTP errors

            res = response.json()
            # print(f"OpenRouter response keys: {list(res.keys())}") # Optional

            if not res.get("choices") or len(res["choices"]) == 0:
                return "" # Return empty if no choices received

            choice = res["choices"][0]
            message = choice.get("message", {})
            citations = res.get("citations") or [] # Get citations if present

            # print(f"Message keys: {list(message.keys())}") # Optional

            content = message.get("content") # Get content, might be None
            reasoning = message.get("reasoning") # Get reasoning, might be None

            # Ensure content and reasoning are strings
            content = content if content is not None else ""
            reasoning = reasoning if reasoning is not None else ""

            # print(f"Found reasoning: {bool(reasoning)} ({len(reasoning)} chars)") # Optional
            # print(f"Found content: {bool(content)} ({len(content)} chars)") # Optional

            # Insert citation links into the text
            content_with_citations = citation_inserter(content, citations)
            reasoning_with_citations = citation_inserter(reasoning, citations)

            # Format the final citation list using the helper
            citation_list_str = citation_formatter(citations)

            # Combine reasoning (if any), content, and citation list
            final_response = ""
            if reasoning_with_citations:
                final_response += f"<think>\n{reasoning_with_citations}\n</think>\n\n"
            if content_with_citations:
                 final_response += content_with_citations

            # Append the formatted citation list if there's any content or reasoning
            if final_response:
                 final_response += citation_list_str

            return final_response

        # --- Improved Error Handling ---
        except requests.exceptions.Timeout:
            print(f"Error in non_stream_response: Request timed out after {timeout}s.")
            return f"Pipe Error: Request timed out ({timeout}s)"
        except requests.exceptions.HTTPError as e:
            error_msg = f"Pipe Error: API returned HTTP {e.response.status_code}"
            try:
                 error_detail = e.response.json().get("error", {}).get("message", "")
                 if error_detail: error_msg += f": {error_detail}"
            except json.JSONDecodeError:
                 error_msg += f" (Body: {e.response.text[:200]})" # Show start of body if not JSON
            except Exception: pass # Ignore other parsing errors
            print(f"Error in non_stream_response: {error_msg} (URL: {e.request.url})")
            print(f"Request payload that caused error: {json.dumps(payload)}") # Log payload on error
            return error_msg
        except requests.exceptions.RequestException as e:
            print(f"Error in non_stream_response: Request failed: {e}")
            return f"Pipe Error: Network error during request: {e}"
        except json.JSONDecodeError as e:
             print(f"Error in non_stream_response: Failed to decode API response JSON: {e}")
             # Attempt to return raw response if decoding fails
             try:
                 raw_response = response.text
                 return f"Pipe Error: Failed to decode API response. Raw response: {raw_response[:500]}"
             except NameError: # response might not be defined if request failed earlier
                 return f"Pipe Error: Failed to decode API response: {e}"
        except Exception as e:
            print(f"Unexpected error in non_stream_response: {e}")
            traceback.print_exc()
            return f"Pipe Error: Unexpected error processing response: {e}"


    def stream_response(self, url, headers, payload, citation_inserter, citation_formatter, timeout):
        """Handles streaming API requests using a generator."""
        response = None # Initialize response to None
        try:
            # print(f"Sending streaming request to OpenRouter: {json.dumps(payload)[:200]}...") # Optional
            response = requests.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            )
            response.raise_for_status() # Check for initial HTTP errors before streaming

            in_reasoning_state = False
            latest_citations = []
            buffer = "" # Buffer for accumulating text chunks

            # Iterate through Server-Sent Events (SSE)
            for line in response.iter_lines():
                if not line: continue # Skip keep-alive newlines

                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "): continue # Ignore non-data lines

                if line_text == "data: [DONE]":
                    # Stream finished, process any remaining buffer
                    if buffer:
                        yield citation_inserter(buffer, latest_citations)
                        buffer = ""
                    # Append final citation list
                    yield citation_formatter(latest_citations)
                    break # Exit loop cleanly

                try:
                    chunk_data = line_text[6:].strip() # Get data part and remove whitespace
                    if not chunk_data: continue # Skip empty data chunks

                    chunk = json.loads(chunk_data)

                    if "choices" in chunk and chunk["choices"]:
                        choice = chunk["choices"][0]
                        # Update citations if present in the chunk
                        chunk_citations = chunk.get("citations")
                        if chunk_citations is not None:
                             latest_citations = chunk_citations

                        delta = choice.get("delta", {})
                        message = choice.get("message", {}) # Handle non-delta messages too

                        reasoning_text = delta.get("reasoning") if delta else message.get("reasoning")
                        content_text = delta.get("content") if delta else message.get("content")

                        # Ensure None is treated as empty string
                        reasoning_text = reasoning_text if reasoning_text is not None else ""
                        content_text = content_text if content_text is not None else ""

                        # Handle reasoning tokens
                        if reasoning_text:
                            if not in_reasoning_state:
                                # Process buffer (content) before starting <think> tag
                                if buffer:
                                    yield citation_inserter(buffer, latest_citations)
                                    buffer = ""
                                yield "<think>\n"
                                in_reasoning_state = True
                            buffer += reasoning_text # Append reasoning to buffer

                        # Handle content tokens
                        if content_text:
                            if in_reasoning_state:
                                # Process buffer (reasoning) before closing </think> tag
                                if buffer:
                                     yield citation_inserter(buffer, latest_citations)
                                     buffer = ""
                                yield "\n</think>\n\n"
                                in_reasoning_state = False
                            buffer += content_text # Append content to buffer

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in stream: {line_text}")
                    continue # Skip this line and continue streaming
                except Exception as e:
                    # Log error during chunk processing but try to continue streaming
                    print(f"Error processing stream chunk: {e} - Line: {line_text}")
                    traceback.print_exc()
                    # Optionally yield an inline error message (might disrupt flow)
                    # yield f"\n[Pipe Warning: Error processing part of the stream: {e}]\n"
                    continue

            # Final check: If stream ended while in reasoning state, close tag and process buffer
            if in_reasoning_state:
                 if buffer:
                     yield citation_inserter(buffer, latest_citations) # Process remaining reasoning
                 yield "\n</think>\n\n" # Close the tag
            elif buffer: # Process any remaining content buffer if not in reasoning state
                 yield citation_inserter(buffer, latest_citations)


        # --- Improved Error Handling ---
        except requests.exceptions.Timeout:
            print(f"Error in stream_response: Request timed out after {timeout}s.")
            yield f"Pipe Error: Request timed out ({timeout}s)"
        except requests.exceptions.HTTPError as e:
            error_msg = f"Pipe Error: API returned HTTP {e.response.status_code}"
            try:
                 # Try reading the response body *after* the exception
                 error_body = e.response.text
                 print(f"HTTPError Body: {error_body[:500]}") # Log error body
                 error_detail = json.loads(error_body).get("error", {}).get("message", "")
                 if error_detail: error_msg += f": {error_detail}"
                 else: error_msg += f" (Body: {error_body[:200]})"
            except json.JSONDecodeError:
                 error_msg += f" (Body: {e.response.text[:200]})" # Show start of body if not JSON
            except Exception as parse_err:
                 print(f"Could not parse HTTPError body: {parse_err}")
                 error_msg += f" (Could not parse error body)"
            print(f"Error in stream_response: {error_msg} (URL: {e.request.url})")
            print(f"Request payload that caused error: {json.dumps(payload)}") # Log payload on error
            yield error_msg # Yield the formatted error to the UI
        except requests.exceptions.RequestException as e:
            print(f"Error in stream_response: Request failed: {e}")
            yield f"Pipe Error: Network error during streaming request: {e}"
        except Exception as e:
            # Catch unexpected errors during streaming setup or iteration
            print(f"Unexpected error in stream_response: {e}")
            traceback.print_exc()
            yield f"Pipe Error: Unexpected error during streaming: {e}"
        finally:
             # Ensure the response connection is closed if it was opened
             if response:
                 response.close()
