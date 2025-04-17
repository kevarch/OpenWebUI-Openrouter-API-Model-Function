# OpenWebUI-Openrouter-API-Model-Function

This pipe integrates the [OpenRouter.ai](https://openrouter.ai/) API with [OpenWebUI](https://openwebui.com/), allowing users to access a wide variety of large language models available through OpenRouter directly within the OpenWebUI interface.

It acts as a bridge, fetching the list of available models from OpenRouter and forwarding chat requests from OpenWebUI to the selected OpenRouter model, handling both streaming and non-streaming responses.

Getting started with OpenRouter is straightforward: simply sign up on their website at [OpenRouter.ai](https://openrouter.ai/), add some credits, and generate an API key to use with this function.

## Features

* **Access to Diverse Models:** Leverages the OpenRouter API to provide access to numerous LLMs from various providers.
* **Model Discovery:** Automatically fetches and lists available OpenRouter models within OpenWebUI. 
    * New models will appear in OpenWebUI as new models are released on OpenRouter.
    * Optionally specify a list of providers to include or exclude from the available OpenRouter models. This is useful if you want to exclude providers you already have integrated in OpenWebUI, such as OpenAI, or only wish to add specific providers.
* **Optional Model Prefix:** Allows adding a custom prefix (e.g., "OR: ") to model names in the OpenWebUI list for easy identification.
* **Streaming & Non-Streaming Support:** Handles both response types seamlessly.
* **Citation Support:**
    * Fetches citation data provided by compatible OpenRouter models.
    * Formats citations into a readable list appended to the response.
    * Replaces citation markers (e.g., `[1]`) in the model's text with clickable Markdown links (e.g., `[[1]](citation_url)`).
* **Reasoning Token Support:**
    * Optionally requests reasoning/thought processes from models that support this feature (`include_reasoning=true`).
    * Wraps reasoning output in `<think>...</think>` tags for clear separation in the UI.
* **Configurable Timeout:** Set a custom timeout for API requests to OpenRouter.
* **Error Handling:** Provides informative error messages in the UI for common issues like invalid API keys, network timeouts, or API errors.

## Installation

This function can be installed manually or easily installed from the OpenWebUI Hub:
[https://openwebui.com/f/preswest/openrouter_integration_for_openwebui](https://openwebui.com/f/preswest/openrouter_integration_for_openwebui)

After installing:

* **Configure the function:**
    * Log in to OpenWebUI as an admin user.
    * Navigate to **Admin Panel** -> **Functions**.
    * Click the **Settings** (gear icon ⚙️) button for the function.
    * Enter your **OpenRouter API Key** (required).
    * Adjust other settings like **Include Reasoning**, **Model Prefix**, and **Request Timeout** as desired.
    * Click **Save**.

## Configuration

The following settings can be configured via the OpenWebUI Pipe settings interface:

* **`OPENROUTER_API_KEY`** (Required): Your personal API key from OpenRouter.ai.
* **`INCLUDE_REASONING`** (Default: `True`): Whether to request reasoning tokens from models that support them. If enabled, reasoning appears within `<think>` tags.
* **`MODEL_PREFIX`** (Default: `""`): An optional text prefix added to the names of OpenRouter models displayed in OpenWebUI (e.g., setting it to `OR: ` will show models like `OR: xAI: Grok 3 Beta`).
* **`REQUEST_TIMEOUT`** (Default: `90`): The maximum time (in seconds) to wait for a response from the OpenRouter API before timing out.
* **`MODEL_PROVIDERS`** Optional list of comma-separated model providers to include or exclude. Leave empty to include all providers.
* **`INVERT_PROVIDER_LIST`** If true, the 'Model Providers' list becomes an *exclude* list instead of an *include* list.

## Usage

1.  After installation and configuration, the models fetched from OpenRouter (potentially with your prefix) will appear in the model selection dropdown in OpenWebUI.
2.  Select an OpenRouter model.
3.  Chat as usual. The pipe will handle communication with the OpenRouter API.
4.  If the model provides citations or reasoning (and reasoning is enabled), they will be formatted and included in the response.

## License

This project is licensed under the MIT License.
