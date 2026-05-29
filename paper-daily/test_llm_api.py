import os

import requests


def main() -> None:
    api_base = os.environ["OPENAI_API_BASE"].rstrip("/")
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ["OPENAI_MODEL_NAME"]

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with exactly one word: ok"},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    print("Status:", response.status_code)
    print("URL:", url)

    try:
        data = response.json()
    except ValueError:
        print("Raw response:", response.text)
        response.raise_for_status()
        return

    print("Response JSON:", data)
    try:
        print("Content:", data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        print("Content: <missing choices[0].message.content>")

    response.raise_for_status()


if __name__ == "__main__":
    main()
