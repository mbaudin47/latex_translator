"""
A tool to translate a LaTeX text from french to english.

- Write the french LaTeX code.
- Push the "Translate button"
- Get the output in english LaTeX code.

## Features

- Drag and drop your LaTeX file
- Select the LLM among llama, deepseek, etc.
- Set the AI temperature: from determistic 0 to fully random 1.
- Enable/disable LaTeX mode
- Select translation tone : Academic, Talk, Concise, etc.
- Set keywords, abstract and difficult terms
- Copy the translated LaTeX term
- Download the translated LaTeX term
- Print the number of tokens, the finish reason, the elapsed time
"""
import streamlit as st
import re
import os
from openai import OpenAI
from groq import Groq
from typing import List, Tuple
import httpx
import re
import time

# Configuration de la page Streamlit
st.set_page_config(page_title="Traducteur LaTeX FR→EN", page_icon="📄", layout="wide")


class LaTeXRawTranslator:
    def __init__(
        self,
        use_groq=True,
        model="llama3-8b-8192",
        latex_mode=False,
        temperature=0.3,
        tone_description="",
        keywords_list=[],
        abstract_text="",
        difficult_terms_dict=dict(),
        must_clean_llm_output=True,
    ):
        """Initialize the LaTeXRawTranslator.

        This class pushes the content to the AI, then ask it to translate.

        Parameters
        ----------
        use_groq : bool, optional
            Whether to use the Groq API.
            Otherwise, use OpenAI.
            Defaults to True.
        model : str, optional
            The model to use for translation. Defaults to "llama3-8b-8192".
        latex_mode : bool, optional
            Whether to enable LaTeX mode. Defaults to False.
        tone_description : str, optional
            The description of the tone of the translation
        keywords_list : list, optional
            The list of keywords
        abstract_text : str, optional
            The abstract
        difficult_terms_dict : dict, optional
            The dictionary of difficult terms
        must_clean_llm_output : bool, optional
            Clean the LLM output if necessary
        """
        if use_groq:
            self.client = Groq(
                # This is the default and can be omitted
                api_key=os.environ.get("GROQ_API_KEY"),
                http_client=httpx.Client(
                    proxy=os.environ.get("HTTP_PROXY"),
                    verify=False,  # or path to your CA bundle
                ),
            )
        else:
            self.client = OpenAI()
            # Use model = "gpt-3.5-turbo" with OpenAI

        self.model = model
        self.latex_mode = latex_mode
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError(
                f"Temperature must be in [0, 1], but temperature={temperature}"
            )
        self.temperature = temperature
        self.tone_description = tone_description
        self.keywords_list = keywords_list
        self.abstract_text = abstract_text
        self.difficult_terms_dict = difficult_terms_dict
        self.must_clean_llm_output = must_clean_llm_output

    def clean_llm_output(self, text: str) -> str:
        """
        Removes the phrase "Here is the translation:" from the beginning of a string.

        Args:
            text: The input string, which may contain the phrase.

        Returns:
            The cleaned string, with the phrase removed if it was present at the start.
        """
        if self.must_clean_llm_output:
            # Use str.startswith() for an efficient check.
            # For Llama
            for phrase_to_remove in [
                "Here is the translation:",
                "Here is the translated text:",
            ]:
                if text.strip().startswith(phrase_to_remove):
                    # Remove the phrase and any leading whitespace that might be left.
                    text = text.strip()[len(phrase_to_remove) :].strip()

            # Cleanup if necessary
            # For Qwen
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text

    def set_tone(self, tone_description):
        """Set the tone description.

        Parameters
        ----------
        tone_description : str
            Description of the desired tone for the translation.
        """
        self.tone_description = tone_description

    def set_keywords(self, keywords_list):
        """Set the keywords list.

        Parameters
        ----------
        keywords_list : list of str
            List of keywords related to the text.
        """
        self.keywords_list = keywords_list

    def set_abstract(self, abstract_text):
        """Set the abstract text.

        Parameters
        ----------
        abstract_text : str
            Summary or abstract of the document.
        """
        self.abstract_text = abstract_text

    def set_difficult_terms_dict(self, difficult_terms_dict):
        """Set the dictionary of difficult terms.

        Parameters
        ----------
        difficult_terms_dict : dict
            Mapping of complex terms to their simplified explanations.
        """
        self.difficult_terms_dict = difficult_terms_dict

    def get_prompt(self) -> str:
        """Generate the prompt instruction string.

            Constructs a detailed instruction set for a scientific translator,
            incorporating tone, keywords, abstract, and difficult terms.

            Returns
            -------
            str
                The generated prompt instructions.
            """
        prompt_instructions = (
                "- You are a professional scientific translator. \n"
            "- Keep punctuation as is. \n"
            "- Keep carriage returns as is. \n"
            "- Do not print: 'Here is the translation:'. \n"
        )
        if self.latex_mode:
            prompt_instructions += (
                "- You are a professional scientific translator. \n"
                "- Preserve any LaTeX syntax intact. \n"
                "- Translate only the content, not the LaTeX commands. \n"
                "- Write a consistent LaTeX code. \n"
                "- Do not translate the comments, indicated by '%' LaTeX command. \n"
                "- Do not modify commands such as : `\\label{}`, `\\ref{}`, `\\cite{}`, etc. \n"
                "- Translate the content of the formatting commands such as : `\\emph{}`, `\\textbf{}`, `\\section{}`, `\\caption{}`, etc. \n"
            )
        if len(self.tone_description) > 0:
            prompt_instructions += f"- {self.tone_description}\n"
        if self.keywords_list:
            keywords_str = ", ".join(self.keywords_list)
            prompt_instructions += (
                f"- The text is related to the following keywords: {keywords_str}.\n"
            )
        if self.abstract_text:
            prompt_instructions += f"- The abstract of the document is as follows:\n\n"
            prompt_instructions += f"{self.abstract_text}\n"
        if self.difficult_terms_dict:
            terms_str = "\n".join(
                [
                    f"- '{f}' should be translated as '{e}'\n"
                    for f, e in self.difficult_terms_dict.items()
                ]
            )
            prompt_instructions += f"- Pay special attention to the following terms and use their provided translations:\n"
            prompt_instructions += f"{terms_str}\n"
        return prompt_instructions

    def translate(self, latex_content: str) -> Tuple[str, int, str]:
        """Translate a complete LaTeX document.

        Parameters
        ----------
        latex_content : str
            The LaTeX content to translate.

        Returns
        -------
        traduction : str
            The translated text.
        total_tokens : int
            The number of token used
        finish_reason : str
            The reason of finishing the AI job.
        """

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text(f"Traduction en cours...")

        try:
            prompt_instructions = self.get_prompt()
            print(f"prompt_instructions:\n{prompt_instructions}")
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_instructions,
                    },
                    {"role": "user", "content": f"Here is the text: {latex_content}"},
                ],
                temperature=self.temperature,
            )
            translated_text = chat_completion.choices[0].message.content
            total_tokens = chat_completion.usage.total_tokens
            finish_reason = chat_completion.choices[0].finish_reason
            translated_text = self.clean_llm_output(translated_text)
        except Exception as e:
            st.error(f"Erreur lors de la traduction : {str(e)}")
            translated_text = latex_content
            finish_reason = "Erreur"
            total_tokens = 0

        progress_bar.empty()
        status_text.empty()

        return translated_text, total_tokens, finish_reason


def main():
    """Main function to run the LaTeX French-to-English translator Streamlit app.

    Sets up the Streamlit UI, handles user input, and manages the translation
    process with configurable parameters such as translation tone, keywords,
    abstract, and difficult terms.
    """
    st.title("🔄 Traducteur LaTeX Français → Anglais")
    st.markdown(
        """
    Cette application traduit vos documents LaTeX du français vers l'anglais tout en préservant 
    la structure et les commandes LaTeX importantes.
    """
    )

    # LLM model selection
    groq_model_descriptions = {
        "llama3-70b-8192": "✅ Excellent for LaTeX and French→English translation — accurate and robust.",
        "deepseek-r1-distill-llama-70b": "🧠 Strong contender — preserves LaTeX structure well and translates accurately.",
        "qwen/qwen3-32b": "✅ Very good multilingual model — handles French well, but may be slightly verbose in output.",
        "llama-3.3-70b-versatile": "🆕 Likely strong in both LaTeX and translation — needs empirical testing.",
        "llama3-8b-8192": "⚠️ Fast and decent — good with LaTeX structure, but less precise than 70B.",
        "gemma2-9b-it": "❌ Limited multilingual and formatting capabilities — often struggles with LaTeX and French.",
    }
    model_names = list(groq_model_descriptions.keys())

    # Define the tone
    translation_tones = {
        "Academic": "Maintain a formal and academic tone suitable for a peer-reviewed journal.",
        "Talk": "Use a tone that is suitable for an academic seminar, lecture, or presentation.",
        "Course": "Employ a clear and simplified tone, as if explaining complex concepts in a textbook or educational course.",
        "Textbook": "Employ a clear and simplified tone, as if explaining complex concepts in an educational textbook or course. Prioritize clarity and pedagogical value.",
        "Concise": "Translate the text with a focus on brevity and clarity, summarizing where appropriate while retaining key scientific points.",
    }
    translation_tones_names = list(translation_tones.keys())

    # Set default parameters
    default_model_name = model_names[0]
    default_temperature = 0.7
    default_latex_mode = True
    default_translation_tone = translation_tones_names[0]
    default_keywords_input = ""
    default_abstract_input = ""
    default_difficult_terms_input = ""
    default_text_height = 400

    # Initialize session state for advanced parameters
    if "selected_language_model" not in st.session_state:
        st.session_state.selected_language_model = default_model_name
    if "temperature" not in st.session_state:
        st.session_state.temperature = default_temperature
    if "latex_mode" not in st.session_state:
        st.session_state.latex_mode = default_latex_mode
    if "translation_tone" not in st.session_state:
        st.session_state.translation_tone = default_translation_tone
    if "keywords_input" not in st.session_state:
        st.session_state.keywords_input = default_keywords_input
    if "abstract_input" not in st.session_state:
        st.session_state.abstract_input = default_abstract_input
    if "difficult_terms_input" not in st.session_state:
        st.session_state.difficult_terms_input = default_difficult_terms_input

    # Interface utilisateur
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Document LaTeX source")

        # Zone de texte pour saisie directe
        latex_input = st.text_area(
            "Collez votre code LaTeX ici :",
            height=default_text_height,
            placeholder="\\documentclass{article}\n\\begin{document}\n\\section{Introduction}\nVotre texte en français ici...\n\\end{document}",
        )

        # Option d'upload de fichier
        uploaded_file = st.file_uploader(
            "ou choisissez un fichier .tex",
            type=["tex"],
            help="Charger votre fichier LaTeX à traduire",
        )

        # Utiliser le contenu du fichier uploadé si disponible
        if uploaded_file is not None:
            latex_content = uploaded_file.read().decode("utf-8")
            st.text_area(
                "Contenu du fichier :", value=latex_content, height=200, disabled=True
            )
        else:
            latex_content = latex_input

        # 🧩 Advanced Parameters Toggle
        show_advanced = st.checkbox("⚙️ Afficher les paramètres avancés", value=False)

        if show_advanced:
            # Paramètres avancés
            st.session_state.selected_language_model = st.selectbox(
                "Sélectionner le modèle :",
                options=model_names,
                index=model_names.index(st.session_state.selected_language_model),
            )
            _ = st.info(
                groq_model_descriptions[st.session_state.selected_language_model]
            )

            # 🌡️ Temperature selector
            st.session_state.temperature = st.slider(
                "🌡️ Modèle de température (0.0 à 1.0)",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                value=st.session_state.temperature,
                help="Plus la température est élevée, plus la traduction est créative (mais moins précise).",
            )
            # Determine the property based on the temperature value
            if 0.0 <= st.session_state.temperature <= 0.5:
                temperature_description = "🧊 Deterministic — always the same answer for the same prompt_instructions."
            elif 0.5 < st.session_state.temperature <= 0.9:
                temperature_description = (
                    "🌤️ Balanced randomness — slightly varied, still reliable."
                )
            elif 0.9 < st.session_state.temperature <= 1.0:
                temperature_description = (
                    "🔥 Creative — more variation and expressive outputs."
                )
            else:
                temperature_description = "🎲 Highly random — responses may become incoherent or unpredictable."

            st.info(temperature_description)

            # 🔘 Checkbox pour activer/désactiver le mode LaTeX
            st.session_state.latex_mode = st.checkbox(
                "📐 Activer le mode LaTeX", value=st.session_state.latex_mode
            )

            # Tone selector!
            st.session_state.translation_tone = st.selectbox(
                "Select Translation Tone:",
                options=translation_tones_names,
                index=translation_tones_names.index(st.session_state.translation_tone),
                help="Choose the style of the translated text.",
            )
            st.info(translation_tones[st.session_state.translation_tone])

            # Keywords
            st.session_state.keywords_input = st.text_input(
                "Mots-clés (séparés par des virgules) :",
                value=(
                    st.session_state.keywords_input
                    if "keywords_input" in st.session_state
                    else ""
                ),
                help="Fournissez des mots-clés spécifiques pour orienter la traduction vers un domaine précis (par ex., 'cosmologie, mécanique quantique').",
            )
            # Abstract
            st.session_state.abstract_input = st.text_area(
                "Abstract du document (optionnel) :",
                height=150,
                value=(
                    st.session_state.abstract_input
                    if "abstract_input" in st.session_state
                    else ""
                ),
                placeholder="Collez ici l'abstract du document pour aider le traducteur à comprendre le contexte...",
                help="Fournir l'abstract du document donne au traducteur un aperçu global du sujet et améliore la précision des termes techniques.",
            )
            # Difficult terms
            st.session_state.difficult_terms_input = st.text_area(
                "Termes difficiles (optionnel) :",
                height=150,
                value=(
                    st.session_state.difficult_terms_input
                    if "difficult_terms_input" in st.session_state
                    else ""
                ),
                placeholder="Entrez les termes difficiles et leur traduction préférée, un par ligne.\nExemple :\n'Apprentissage profond' -> 'Deep Learning'\n'Réseau de neurones' -> 'Neural network'",
                help="Fournissez une liste de termes techniques spécifiques à traduire de manière précise. Utilisez le format 'Terme français' -> 'Terme anglais'.",
            )
        else:
            # Display current values in read-only mode
            short_parameters_description = (
                f"Modèle : {st.session_state.selected_language_model}, "
                f"Température : {st.session_state.temperature}, "
                f"Mode LaTeX : {st.session_state.latex_mode}, "
                f"Ton: {st.session_state.translation_tone}"
            )
            if st.session_state.keywords_input:
                short_parameters_description += (
                    f", Mots-clés: {st.session_state.keywords_input}"
                )
            if st.session_state.abstract_input:
                short_parameters_description += (
                    f", Résumé: {st.session_state.abstract_input[:10]}..."
                )
            if st.session_state.difficult_terms_input:
                short_parameters_description += f", Termes difficiles: {st.session_state.difficult_terms_input[:10]}..."

            st.text(short_parameters_description)

    with col2:
        st.subheader("🔄 Document traduit")

        if st.button("🚀 Traduire", type="primary", use_container_width=True):
            if latex_content.strip():
                with st.spinner("Traduction en cours..."):
                    try:
                        start_time = time.time()
                        # Initialiser le traducteur
                        translator = LaTeXRawTranslator(
                            model=st.session_state.selected_language_model,
                            latex_mode=st.session_state.latex_mode,
                            temperature=st.session_state.temperature,
                        )
                        # Get the selected tone description from the session state
                        tone_description = translation_tones[
                            st.session_state.translation_tone
                        ]
                        translator.set_tone(tone_description)

                        # Get the keywords
                        # In your translation function, after getting the input
                        keywords_list = [
                            keyword.strip()
                            for keyword in st.session_state.keywords_input.split(",")
                            if keyword.strip()
                        ]
                        translator.set_keywords(keywords_list)

                        # Get the abstract
                        abstract_text = st.session_state.abstract_input
                        translator.set_abstract(abstract_text)

                        # In your translation function, after getting the input
                        difficult_terms_dict = {}
                        if st.session_state.difficult_terms_input:
                            lines = (
                                st.session_state.difficult_terms_input.strip().split(
                                    "\n"
                                )
                            )
                            for line in lines:
                                if "->" in line:
                                    try:
                                        french_term, english_term = line.split("->", 1)
                                        difficult_terms_dict[french_term.strip()] = (
                                            english_term.strip()
                                        )
                                    except ValueError:
                                        # Handle malformed lines gracefully
                                        continue
                        translator.set_difficult_terms_dict(difficult_terms_dict)

                        # Translate
                        translated_content, total_tokens, finish_reason = (
                            translator.translate(latex_content)
                        )
                        prompt_instructions = translator.get_prompt()
                        duration = time.time() - start_time

                        # Afficher le résultat
                        if st.session_state.latex_mode:
                            st.code(
                                translated_content,
                                height=default_text_height,
                                language="latex",
                            )
                        else:
                            st.text_area(
                                "Résultat traduit :",
                                value=translated_content,
                                height=default_text_height,
                            )
                        # ✅ Affichage des tokens utilisés
                        st.info(f"🔢 Tokens utilisés : {total_tokens}")
                        st.info(f"ℹ️ Raison de terminaison : {finish_reason}")
                        st.info(f"🔢 Durée : {duration:.2f} (s)")

                        # Bouton de téléchargement
                        st.download_button(
                            label="📥 Télécharger le fichier traduit",
                            data=translated_content,
                            file_name="document_traduit.tex",
                            mime="text/plain",
                        )

                        st.success("✅ Traduction terminée avec succès !")
                        with st.expander("ℹ️ prompt_instructions"):
                            st.markdown(prompt_instructions)

                    except Exception as e:
                        st.error(f"❌ Erreur lors de la traduction : {str(e)}")
            else:
                st.warning("⚠️ Veuillez fournir du contenu LaTeX à traduire.")

    # Section d'aide
    with st.expander("ℹ️ Aide et exemples"):
        st.markdown(
            """
        ### Exemples de commandes gérées :
        
        **Préservées (non traduites) :**
        - `\\label{eq:example}` → `\\label{eq:example}`
        - `\\ref{fig:graph}` → `\\ref{fig:graph}`
        - `\\cite{author2023}` → `\\cite{author2023}`
        
        **Traduites :**
        - `\\emph{texte important}` → `\\emph{important text}`
        - `\\section{Sur les processus stochastiques}` → `\\section{On stochastic processes}`
        - `\\caption{Description de la figure}` → `\\caption{Figure description}`
        
        ### Conseils d'utilisation :
        - Vérifiez toujours le résultat avant utilisation
        - Les formules mathématiques sont automatiquement préservées
        - Les commentaires LaTeX (%) sont préservés
        """
        )


if __name__ == "__main__":
    main()
