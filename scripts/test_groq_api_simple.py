import os
from groq import Groq
import httpx

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),  
    http_client=httpx.Client(
        proxy=os.environ.get("HTTP_PROXY"),
        verify=False  # or path to your CA bundle
    )
)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="llama3-8b-8192",
)
print(chat_completion.choices[0].message.content)
