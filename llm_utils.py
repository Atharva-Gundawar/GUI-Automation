import requests
import os
from image_utils import encode_image

# Constants
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

claude_headers = {
    "Content-Type": "application/json",
    "X-API-Key": CLAUDE_API_KEY,
    "anthropic-version": "2023-06-01"
}

openai_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def segment_ranker(base64_images, user_query):
    """Creates the API request payload for image ranking based on user query."""
    
    text = f"Return the index (starting from 0) of the image that most closely resembles a {user_query}.\nAnswer with only a single number."

    # Create the initial message structure with text content
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": text}]
    }]

    # Add each base64 image to the message content
    for _, base64_image in base64_images:
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
        messages[0]['content'].append(image_content)

    # Create the complete API request payload
    return {
        "model": 'gpt-4o',
        "messages": messages,
        "max_tokens": 300
    }

def make_history(elements):
    """Formats a history of actions into a string."""
    output = "History of action: (includes any actions taken in the past):\n"
    for i, element in enumerate(elements, 1):
        output += f"{i}.\n" + "{\n"
        output += ''.join(f"    '{key}':'{value}',\n" for key, value in element.items())
        output = output.rstrip(',\n') + "\n}\n\n"
    return output

def ask_claude(image_path, question):
    """Sends a request to the Claude API with an image and a question."""
    
    base64_image = encode_image(image_path)

    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }]
    }

    response = requests.post(CLAUDE_API_URL, json=payload, headers=claude_headers)

    if response.status_code == 200:
        return response.json()['content'][0]['text']
    return f"Error: {response.status_code}, {response.text}"

def ask_gpt4(image_input, question, text_input=False):
    """Sends a request to the GPT-4 API with an image or text and a question."""
    
    if text_input:
        image_input = encode_image(image_input)
        
    payload = {
        "model": "gpt-4-turbo",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_input}"}
                }
            ]
        }],
        "max_tokens": 300
    }

    response = requests.post(OPENAI_API_URL, json=payload, headers=openai_headers)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    image_path = "screenshot.png"
    question = "What do you see in this image?"

    claude_answer = ask_claude(image_path, question)
    print(claude_answer)

    gpt4_answer = ask_gpt4(image_path, question, text_input=True)
    print(gpt4_answer)
