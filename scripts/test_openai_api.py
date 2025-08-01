import os
import openai
from groq import Groq

# Récupère la clé depuis la variable d’environnement
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("La variable d'environnement OPENAI_API_KEY n'est pas définie.")

# Initialise l'API
openai.api_key = api_key

client = openai.OpenAI()


try:
    # Appel simple à l'API Chat (GPT-4 ou GPT-3.5 selon ton abonnement)
    response = openai.chat.completions.create(
        model="gpt-4",  # Remplace par "gpt-3.5-turbo" si tu n'as pas accès à GPT-4
        messages=[
            {"role": "system", "content": "Tu es un assistant utile."},
            {"role": "user", "content": "Donne-moi une citation inspirante en français."}
        ]
    )

    print("✅ Réponse de l'API OpenAI :")
    print(response.choices[0].message["content"])

except Exception as e:
    print("❌ Erreur lors de l'appel à l'API :", e)
