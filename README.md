# OpenWebUI-Openrouter-API-Model-Function

This pipe integrates the [OpenRouter.ai](https://openrouter.ai/) API with [OpenWebUI](https://openwebui.com/), allowing users to access a wide variety of large language models available through OpenRouter directly within the OpenWebUI interface.

It acts as a bridge, fetching the list of available models from OpenRouter and forwarding chat requests from OpenWebUI to the selected OpenRouter model, handling both streaming and non-streaming responses.

Getting started with OpenRouter is straightforward: simply sign up on their website at [OpenRouter.ai](https://openrouter.ai/), add some credits, and generate an API key to use with this function.

## Features

*   **Access to Diverse Models:** Leverages the OpenRouter API to provide access to numerous LLMs from various providers.
*   **Model Discovery:** Automatically fetches and lists available OpenRouter models within OpenWebUI.
    *   New models will appear automatically as they are released on OpenRouter.
    *   Optionally specify a list of providers to include or exclude, useful to avoid duplication with models already integrated in OpenWebUI or to focus on specific providers.
*   **Model Whitelist (`MODEL_WHITELIST`):** Optionally define a comma-separated list of specific OpenRouter model IDs to include. If set, only these models will be available, overriding provider filtering.
*   **Pricing Display (`SHOW_PRICING`):** Optionally display pricing information (input cost, output cost, total cost per 1 Million tokens) directly in the model names for better cost awareness.
*   **Optional Model Prefix:** Allows adding a custom prefix (e.g., `OR: `) to model names in the OpenWebUI list for easy identification. This field is optional and can be left empty for more flexible configuration.
*   **Free Models Filter (`FREE_ONLY`):** Enables showing only free models to facilitate testing and usage without consuming credits.
*   **Streaming & Non-Streaming Support:** Handles both response types seamlessly.
*   **Citation Support:**
    *   Fetches citation data provided by compatible OpenRouter models.
    *   Formats citations into a readable list appended at the end of the response.
    *   Replaces citation markers (e.g., `[1]`) with clickable Markdown links (e.g., `[[1]](citation_url)`).
*   **Reasoning Token Support:**
    *   Optionally requests reasoning/thought processes from models that support this feature.
    *   **Configurable Effort Levels (`REASONING_EFFORT`):** Set a default reasoning effort to "high" (~80% tokens), "medium" (~50% tokens), or "low" (~20% tokens).
    *   **Dynamic Effort Detection:** Override the default effort by starting your user message with "high", "medium", or "low". The keyword will be automatically removed from the message sent to the model.
    *   Wraps reasoning output in `<think>...</think>` tags for clear separation in the UI.
    *   Provides a clear Markdown notification (e.g., `> *Reasoning set to high*`) at the beginning of the response when dynamic effort detection is used.
*   **Configurable Timeout:** Set a custom timeout for API requests to OpenRouter.
*   **Error Handling:** Provides informative error messages in the UI for common issues like invalid API keys, network timeouts, or API errors.
*   **Optional Prompt Caching:** Allows enabling OpenRouter's prompt caching feature (`cache_control`), potentially reducing costs for supported models on repeated requests with similar initial prompts.
*   **Optional Model Providers (`MODEL_PROVIDERS`):** This parameter accepts a comma-separated list of providers to include or exclude and is optional for more flexible setup.
*   **Per-Model Provider Blacklist (`MODEL_PROVIDER_BLACKLIST`):** Allows you to exclude specific providers for certain models. See Configuration for details.

## Installation

This function can be installed manually or easily installed from the OpenWebUI Hub:
[https://openwebui.com/f/preswest/openrouter_integration_for_openwebui](https://openwebui.com/f/preswest/openrouter_integration_for_openwebui)

After installing:

*   **Configure the function:**
    *   Log in to OpenWebUI as an admin user.
    *   Navigate to **Admin Panel** -> **Functions**.
    *   Click the **Settings** (gear icon ⚙️) button for the function.
    *   Enter your **OpenRouter API Key** (required).
    *   Adjust other settings such as **Include Reasoning**, **Reasoning Effort**, **Model Prefix**, **Request Timeout**, **FREE_ONLY**, **Model Whitelist**, and **Show Pricing** as desired.
    *   Click **Save**.

## Configuration

The following settings can be configured via the OpenWebUI Pipe settings interface:

*   **`OPENROUTER_API_KEY`** (Required): Your personal API key from OpenRouter.ai.
*   **`INCLUDE_REASONING`** (Default: `True`): Whether to request reasoning tokens from models that support them. If enabled, reasoning appears within `<think>` tags. This setting controls if `REASONING_EFFORT` is applied.
*   **`REASONING_EFFORT`** (Default: `medium`): The default reasoning effort level to request from the model. Valid options are `high` (~80% tokens for reasoning), `medium` (~50% tokens), and `low` (~20% tokens). This can be dynamically overridden by starting your user message with "high", "medium", or "low".
*   **`MODEL_PREFIX`** (Optional, Default: `""`): An optional text prefix added to the names of OpenRouter models displayed in OpenWebUI (e.g., setting it to `OR: ` will show models like `OR: xAI: Grok 3 Beta`).
*   **`REQUEST_TIMEOUT`** (Default: `90`): The maximum time (in seconds) to wait for a response from the OpenRouter API before timing out.
*   **`MODEL_PROVIDERS`** (Optional): Comma-separated list of model providers to include or exclude. Leave empty to include all providers.
*   **`INVERT_PROVIDER_LIST`**: If true, the 'Model Providers' list becomes an *exclude* list instead of an *include* list.
*   **`MODEL_WHITELIST`** (Optional): Comma-separated list of specific model IDs (e.g., `anthropic/claude-3-sonnet,openai/gpt-4`). If set, only these models will be available, overriding provider filtering.
*   **`MODEL_PROVIDER_BLACKLIST`** (Optional): Allows you to exclude specific providers for certain models. Format: `'model1:provider1,provider2;model2:provider3'`. For example, `'gpt-4:openai;claude:anthropic,aws'` will exclude OpenAI's GPT-4 and Anthropic/AWS versions of Claude models. This is useful if you want to avoid certain providers for specific models. Leave empty to disable this filter.
*   **`ENABLE_CACHE_CONTROL`** (Default: `False`): If true, enables OpenRouter's prompt caching, potentially reducing costs for supported models on repeated requests with similar initial prompts and repeated context.
*   **`FREE_ONLY`** (Optional, Default: `False`): If true, filters to show only free models.
*   **`SHOW_PRICING`** (Optional, Default: `True`): If true, displays pricing information (input, output, total cost per 1M tokens) in the model names.

## Usage

1.  After installation and configuration, the models fetched from OpenRouter (potentially with your prefix and pricing info) will appear in the model selection dropdown in OpenWebUI.
2.  If desired, enable the filter for free models with `FREE_ONLY`.
3.  Select an OpenRouter model.
4.  Chat as usual.
    *   If you want to dynamically control reasoning effort for a specific message, start your question with `high`, `medium`, or `low` (e.g., `high How do I calculate orbital mechanics?`).
    *   A notification will appear at the start of the response indicating the set reasoning effort.
5.  The pipe will handle communication with the OpenRouter API.
6.  If the model provides citations or reasoning (and reasoning is enabled), they will be formatted and included in the response.

## License

This project is licensed under the MIT License.