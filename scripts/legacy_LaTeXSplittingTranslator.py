class LaTeXSplittingTranslator:
    def __init__(self, use_groq=True, model="llama3-8b-8192", temperature=0.3,
                 tone_description=""):
        """Initialize the LaTeXSplittingTranslator.

        This class splits the text into LaTeX segments, translate each
        segment, then aggregate the pieces to create the full translation.

        Parameters
        ----------
        use_groq : bool, optional
            Whether to use the Groq API.
            Otherwise, use OpenAI.
            Defaults to True.
        model : str, optional
            The model to use for translation. Defaults to "llama3-8b-8192".
        temperature : float, in [0, 1]
            The temperature of the model.
        tone_description : str, optional
            The description of the tone of the translation
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
        # Commandes LaTeX à préserver (ne pas traduire)
        self.preserve_commands = {
            "label",
            "ref",
            "cite",
            "citep",
            "citet",
            "pageref",
            "eqref",
            "includegraphics",
            "input",
            "include",
            "bibliography",
            "bibliographystyle",
            "newcommand",
            "renewcommand",
            "documentclass",
            "usepackage",
            "begin",
            "end",
            "item",
            "itemize",
            "enumerate",
            "description",
            "label",
            "ref",
            "cite",
            "includegraphics",
            "caption",
            "footnote",
            "mathbb",
            "mathbf",
            "mathrm",
            "mathcal",
            "frac",
            "sqrt",
            "sum",
            "int",
            "lim",
            "infty",
            "left",
            "right",
            "author",
            "marginpar",
        }

        # Commandes LaTeX dont le contenu doit être traduit
        self.translate_commands = {
            "emph",
            "textbf",
            "textit",
            "underline",
            "textsc",
            "textsf",
            "texttt",
            "title",
            "section",
            "subsection",
            "subsubsection",
            "chapter",
            "part",
            "paragraph",
            "subparagraph",
            "caption",
            "footnote",
        }
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError(
                f"Temperature must be in [0, 1], but temperature={temperature}"
            )
        self.temperature = temperature
        self.tone_description = tone_description

    def set_tone(self, tone_description):
        self.tone_description = tone_description

    def extract_latex_segments(self, text: str) -> List[Tuple[str, str, bool]]:
        """Extract segments from LaTeX text.

        Parameters
        ----------
        text : str
            The LaTeX text to process.

        Returns
        -------
        List[Tuple[str, str, bool]]
            A list of tuples containing the segment, its type, and whether to translate.
        """
        segments = []
        i = 0

        while i < len(text):
            if text[i] == "\\":
                # Trouver la fin de la commande
                j = i + 1
                while j < len(text) and text[j].isalpha():
                    j += 1

                command = text[i + 1 : j]

                if j < len(text) and text[j] == "{":
                    # Trouver l'accolade fermante correspondante
                    brace_count = 1
                    k = j + 1
                    while k < len(text) and brace_count > 0:
                        if text[k] == "{":
                            brace_count += 1
                        elif text[k] == "}":
                            brace_count -= 1
                        k += 1

                    if brace_count == 0:
                        # Commande complète trouvée
                        full_command = text[i:k]
                        content = text[j + 1 : k - 1]

                        if command in self.preserve_commands:
                            segments.append((full_command, "preserve", False))
                        elif command in self.translate_commands:
                            segments.append((f"\\{command}{{", "command_start", False))
                            segments.append((content, "translate_content", True))
                            segments.append(("}", "command_end", False))
                        else:
                            segments.append((full_command, "unknown_command", False))

                        i = k
                        continue

                # Commande sans accolades ou malformée
                segments.append((text[i:j], "command_no_braces", False))
                i = j
            else:
                # Texte normal - chercher le prochain backslash ou la fin
                j = i
                while j < len(text) and text[j] != "\\":
                    j += 1

                if j > i:
                    text_segment = text[i:j]
                    # Ne traduire que si ce n'est pas juste des espaces/retours à la ligne
                    if text_segment.strip():
                        segments.append((text_segment, "normal_text", True))
                    else:
                        segments.append((text_segment, "whitespace", False))

                i = j

        return segments

    def translate_segment(
        self, text: str, prompt, use_text_tag=False
    ) -> Tuple[str, int, str]:
        """Translate a segment from French to English using AI.

        Parameters
        ----------
        text : str
            The text to translate.
        prompt : str
            Instruction for the translation.
        use_text_tag : bool, optional
            Whether to wrap the translation in <text> tags. Defaults to False.

        Returns
        -------
        traduction : str
            The translated text.
        total_tokens : int
            The number of token used
        finish_reason : str
            The reason of finishing the AI job.
        """
        if use_text_tag:
            prompt += "Encadre le texte avec des balises <text> ... </text>."
        traduction = text
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {"role": "user", "content": f"Here is the text: {text}"},
                ],
                temperature=self.temperature,
            )
            ai_result = chat_completion.choices[0].message.content
            total_tokens = chat_completion.usage.total_tokens
            finish_reason = chat_completion.choices[0].finish_reason

            print(f"ai_result: {ai_result}")
            if use_text_tag:
                resultat_re = re.search(r"<text>(.*?)</text>", ai_result)
                if traduction:
                    traduction = resultat_re.group(1)
                else:
                    print("> Aucune balise <text> trouvée.")
                    traduction = ai_result
            else:
                traduction = ai_result

        except Exception as e:
            st.error(f"Erreur lors de la traduction : {str(e)}")
            traduction = text
            total_tokens = 0
            finish_reason = "Erreur"

        return traduction, total_tokens, finish_reason

    def get_prompt(self) -> str:
        prompt = (
            "- You are a professional scientific translator. \n"
            "- Translate the following French paragraph into English, but preserve any LaTeX syntax intact. \n"
            "- Translate only the content, not the LaTeX commands. \n"
            "- Keep carriage returns as is. \n"
            "- Do not translate punctuation marks: '.', ',', ';', etc. \n"
            "- Do not translate symbols '+', '-', '*', '/', etc. \n"
            "- Do not print: 'Here is the translation:'. \n"
        )
        if len(self.tone_description) > 0:
            prompt += f"- {self.tone_description}"
        return prompt

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
        segments = self.extract_latex_segments(latex_content)
        translated_segments = []
        prompt = self.get_prompt()
        print(f"prompt:\n{prompt}")

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_tokens = 0
        finish_reason = ""

        for i, (segment, segment_type, should_translate) in enumerate(segments):
            progress = (i + 1) / len(segments)
            progress_bar.progress(progress)
            status_text.text(f"Traduction en cours... {i+1}/{len(segments)} segments")
            print(f"Traduction en cours... {i+1}/{len(segments)} segments")
            print(
                f"Segment: {segment}, Type: {segment_type}, Should translate: {should_translate}"
            )

            if should_translate and segment.strip():
                translated_segment, local_tokens, finish_reason = (
                    self.translate_segment(segment, prompt)
                )
                total_tokens += local_tokens
                translated_segments.append(translated_segment)
                print(f"Traduction: {translated_segment}")
            else:
                translated_segments.append(segment)

        progress_bar.empty()
        status_text.empty()

        translated_text = "".join(translated_segments)
        return translated_text, total_tokens, finish_reason
