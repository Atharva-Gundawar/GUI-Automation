import time
import json
import os
import pyautogui
import webbrowser
import logging
from openai import OpenAI
from llm_utils import make_history, ask_claude, ask_gpt4
from image_utils import encode_image, segment_image, crop_image_from_mask
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
PROMPT_FILE_PATH = "prompts/prompt_main.txt"
SCREENSHOT_PATH = "screenshot.png"
URL = "https://www.amazon.com"
SLEEP_INTERVAL = 5

# Ensure the logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Construct the full path to the log file
log_file_path = os.path.join(log_dir, "app.log")

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(log_file_path, mode='w')  # Logs to a file (overwrites each run)
    ]
)

def read_prompt(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {file_path}")
        return None

def capture_screenshot(save_path=SCREENSHOT_PATH):
    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)
    return encode_image(save_path)

def query_claude_with_context(main_prompt, action_history, user_query):
    action_history_string = make_history(action_history) if action_history else ""
    prompt = main_prompt.replace("_action_history", action_history_string)
    return prompt.replace("_user_query", user_query)

def process_claude_response(response):
    try:
        cleaned_response = response.replace("'", '"').replace('```json', '').replace("```", "")
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return None

def handle_scroll_action():
    pyautogui.scroll(-20)

def handle_non_scroll_action(client, answer_claude, screenshot_base64):
    logging.info("Sending image to Segment Anything model")
    mask_links = segment_image(screenshot_base64)
    logging.info("Cropped images links:\n" + "\n".join(mask_links))
    logging.info(f'In images looking for: {answer_claude["description"]}\n')
    
    visual_query = f'Answer in True or False only. Does this image contain or resembles a {answer_claude["description"]}?'
    
    shortlisted_masks = []

    for _, mask_link in enumerate(mask_links):
        centroid, b64_mask = crop_image_from_mask(SCREENSHOT_PATH, mask_link)
        answer_v_gpt = ask_gpt4(b64_mask, visual_query)
        logging.info(f'Answer from GPT-4 AI: {answer_v_gpt}')
        if 'True' in answer_v_gpt:
            logging.info(f'Point of interest: {centroid}')
            shortlisted_masks.append([centroid, b64_mask])

    if not shortlisted_masks:
        logging.warning("No suitable masks found.")
        return

    logging.info("Choosing the best image from the shortlisted masks")
    response = select_best_mask(client, shortlisted_masks, answer_claude["description"])
    if response is not None:
        point_to_click = shortlisted_masks[response][0]
        logging.info(f"Clicking on the point: {point_to_click}")
        time.sleep(SLEEP_INTERVAL)
        pyautogui.click(int(point_to_click[0]), int(point_to_click[1]))
        time.sleep(SLEEP_INTERVAL)

        if answer_claude["action"] == "click and type":
            pyautogui.typewrite(answer_claude["text"], interval=0.25)
            pyautogui.press('enter')

def select_best_mask(client, shortlisted_masks, description):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"return the index (starting from 0) of the image that most closely resembles a {description}?\nAnswer with only a single number and no more information. Answer:"}
        ]
    }]

    for _, b64_mask in shortlisted_masks:
        messages[0]['content'].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_mask}", "detail": "high"}
        })

    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        response = int(response.choices[0].message.content.strip())
        return response
    except Exception as e:
        logging.error(f"Error selecting the best mask: {e}")
        return None

def main():
    client = OpenAI()
    action_history = []

    main_prompt = read_prompt(PROMPT_FILE_PATH)
    if main_prompt is None:
        return

    user_query = input("Enter your query: ")

    main_prompt = main_prompt.replace("_user_query", user_query)
    logging.info(main_prompt)

    webbrowser.open(URL)

    while True:
        time.sleep(SLEEP_INTERVAL)
        screenshot_base64 = capture_screenshot()

        logging.info("Querying Claude AI with the screenshot and user query")

        main_prompt_with_context = query_claude_with_context(main_prompt, action_history, user_query)
        logging.info(main_prompt_with_context)

        answer_claude = ask_claude(SCREENSHOT_PATH, main_prompt_with_context)
        logging.info("Answer from Claude AI: " + str(answer_claude))

        answer_claude = process_claude_response(answer_claude)
        if answer_claude is None:
            return

        logging.info(f'Action from Claude AI: {answer_claude["action"]}')
        action_history.append(answer_claude)

        if answer_claude["action"] == "scroll":
            handle_scroll_action()
        else:
            handle_non_scroll_action(client, answer_claude, screenshot_base64)

        try:
            os.remove(SCREENSHOT_PATH)
        except OSError as e:
            logging.warning(f"Error removing file: {SCREENSHOT_PATH} - {e}")

if __name__ == "__main__":
    main()
