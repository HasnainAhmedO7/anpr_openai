from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Define the text prompt
prompt = """
Can you extract the vehicle number plate text inside the image?
If you are not able to extract text, please respond with None.  # Fallback instruction
Only output text, please.  # Ensure no extra formatting
If any text character is not from the English language, replace it with a dot (.)  # Handle non-English characters, because OpenCV directly can't process these.
"""


def extract_text(base64_encoded_data):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_data}"},
                    },
                ],
            }
        ],
    )
    return response