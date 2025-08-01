import os
from groq import Groq
import httpx
import re

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),  
    http_client=httpx.Client(
        proxy=os.environ.get("HTTP_PROXY"),
        verify=False  # or path to your CA bundle
    )
)


text_to_translate = "Ceci est un texte important en français. Nous allons tester la traduction."

# Part 1 : using user role only
print("Part 1 : using user role only")
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Tu es un traducteur expert. Traduis le texte français suivant en anglais de manière naturelle et fluide. Préserve la ponctuation et la mise en forme. Ne traduis que le contenu, pas les commandes LaTeX. Encadre le texte avec des balises <text> ... </text>. Voici le texte : {text_to_translate}",
        }
    ],
    model="llama3-8b-8192",
)
ai_result = chat_completion.choices[0].message.content
print(f"ai_result: {ai_result}")

contenu = re.search(r"<text>(.*?)</text>", ai_result)

if contenu:
    traduction = contenu.group(1)
    print(f"Traduction: {traduction}")
else:
    print("Aucune balise <text> trouvée.")

# Part 2 : using system, then user role
print("Part 2 : using system, then user role")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"Tu es un traducteur expert. Traduis le texte français suivant en anglais de manière naturelle et fluide. Préserve la ponctuation et la mise en forme. Ne traduis que le contenu, pas les commandes LaTeX. Encadre le texte avec des balises <text> ... </text>.",
        },
        {
            "role": "user",
            "content": f"Here is the text: {text_to_translate}"
        }
    ],
    model="llama3-8b-8192",
)
ai_result = chat_completion.choices[0].message.content
print(f"ai_result: {ai_result}")

contenu = re.search(r"<text>(.*?)</text>", ai_result)

if contenu:
    traduction = contenu.group(1)
    print(f"Traduction: {traduction}")
else:
    print("Aucune balise <text> trouvée.")
