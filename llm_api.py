import os
from typing import List
from google import genai
from google.genai import types
import pydantic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SubComment(pydantic.BaseModel):
    sub_comment: str
    topic: str
    sentiment: float


# Initialize the client with API key from environment variable
# Will raise error if not found to avoid silent failures
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not found. Please add it to your .env file."
    )

client = genai.Client(api_key=api_key)

model = "gemini-2.0-flash"


def llm_generate(user_comment: str) -> List[SubComment]:
    prompt = f"""
    You are a helpful assistant that can analyze user's comment and split it into a list of sub-comments (size from 1 to 10, 1 means that the sub-comment is the same as the user's comment).
    Each sub-comment should keep the exactly same text in the user's comment. You just hard split the comment into several sub-comments.
    You are not allowed to change the text in the user's comment.
    For each sub-comment, you should also provide a topic (must in ["price", "quality", "design", "function", "customer service", "other"]) related to the sub-comment that the user is talking about, and a sentiment score (from -1 to 1, -1 means that the sub-comment is negative, 1 means that the sub-comment is positive, 0 means that the sub-comment is neutral) to describe the user's attitude towards the sub-comment with the topic.
    Here is an example for user's comment:
    "I love this product, but the price is too high."
    The example output should be (in JSON format):
    [
        {{
            "sub_comment": "I love this product",
            "topic": "quality",
            "sentiment": 1
        }},
        {{
            "sub_comment": ", but the price is too high.",
            "topic": "price",
            "sentiment": -1
        }}
    ]

    Here is other example for user's comment:
    "I can't afford this product, it's too expensive."
    The example output should be:
    [
        {{
            "sub_comment": "I can't afford this product, it's too expensive.",
            "topic": "price",
            "sentiment": -1
        }}
    ]

    NOTE: sub-comment split is hard split, it should can be back to the original comment if we concatenate all sub-comments with empty string (not space). So you need to make sure include the punctuation and space in the sub-comment. As general rule, it is better to let following sub-comment carry the punctuation and space that the previous sub-comment ends with.

    NOTE: If previous or following sub-comment has the same topic, you should merge them into one sub-comment. Because in general, we think those sub-comments are talking about the same thing.
    
    Now, please analyze the user's comment:
    "{user_comment}"
    Please output the result:
    """
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=list[SubComment],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    print(response.text)

    return response.parsed


if __name__ == "__main__":
    llm_generate("The user experience is not good, I can't use it.")
