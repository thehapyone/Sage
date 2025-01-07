import base64
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate, PromptTemplate


@dataclass
class ChatPrompt:
    ai_assistant_name: str = "Sage"

    tool_name: str = "multi_source_inquiry"

    description_pre: str = (
        "A tool for providing detailed and verified information for answering questions that require insights from various data sources. "
        "When to use: "
        "- You need answers that could be possibly found in external sources or documents."
        "- The question could be linked to data from specific, known sources. "
        "- You need answers to something outside your own knowledge. "
        "Input: A clear and concise question. "
        "Capabilities: "
        "- Retrieves and synthesizes data from relevant source database to construct answers. "
        "Example input: "
        "- 'How many team members are in the Xeres Design Team?' "
        "- 'What are the current configurations for the platform test environments?' "
    )

    query_generator_system_prompt: str = """
    You are a query generator. Based on the User Input and Chat History, generate up to 3 concise unique and relevant search queries that relate to the user's request.
    Guidelines:
     - Generate between 1 to 3 queries, depending on the user's input and complexity.
     - Ensure all queries are distinct.
     - Focus on identifying the user's intent (factual, exploratory, conversational) to shape query generation.
     - Do not provide any explanations or additional text.
     - Output the queries in the same format as the examples. NO EXPLANATION

    Examples:
    Input:
     - User Input: "Tell me about the Eiffel Tower."
     - Chat History: ""
    Output: ["Eiffel Tower history", "Eiffel Tower facts", "Eiffel Tower significance"]

    Input:
     - User Input: "What other landmarks are nearby?"
     - Chat History: "Tell me about the Eiffel Tower."
    Output: ["Landmarks near Eiffel Tower", "Nearby attractions to Eiffel Tower", "Paris landmarks close to Eiffel Tower"]

    Input:
     - User Input: "Explain the greenhouse effect."
     - Chat History: "What causes global warming?"
    Output: ["Greenhouse effect explanation", "Role of greenhouse gases in global warming"]
    --------------------------------------------------------
    Now, generate queries for the following:
    """

    query_generator_user_prompt: str = """
    User Input: {question}

    <chat_history>
    {chat_history}
    </chat_history>
    """

    qa_system_prompt = """
    You are Sage, an AI assistant.
    **Your Goal:** Provide accurate, neutral, and relevant answers to the user's questions using the given **Context** and your own knowledge.  

    ### Instructions:
    - **Use Context First:** Incorporate 'context' for accuracy. If insufficient or inrelevant, rely on internal knowledge.
    - **Clarity:** Keep responses concise and avoid redundancy. Use bullet points if helpful.
    - **Neutrality:** Present facts without opinions or assumptions.
    - **Ambiguities:** Avoid guessing meanings for unclear terms or abbreviations.

    ### Citations:
    - If you use information from the **Context**, include a **"Sources"** section at the end of your answer.
    - Format the sources like this:
    Sources:
    [1] - Brief summary of the first source. (Less than 5 words)
    [2] - Brief summary of the second source.
    - Do not include Sources if the conversation is casual or if you didn't use the **Context**.
    """

    qa_user_prompt: str = """
    Question: {question}

    <context>
    {context}
    </context>
    
    Here is the current chat history - use if relevant:
    <chat_history>
    {chat_history}
    </chat_history>
    """

    """The prompt template for the chat complete chain"""

    def tool_description(self, source_repr: str) -> str:
        """
        Generates a description for the source QA tool

        Args:
            source_repr (str): A source metadata in a string representation in strings

        Returns:
            str: A tool description
        """
        source_description = source_repr.replace("\n  ", " ")
        description = f"{self.description_pre}\n. I have access to the following sources: {source_description}"
        return description

    @staticmethod
    def generate_welcome_message(
        greeting: str, source_repr: str, profile: str = "chat"
    ):
        """
        Generate and format an introduction message.

        Args:
            greeting (str): A time of day greeting message
            source_repr (str): A source metadata in a string representation in strings
            profile (str): The chat mode profile

        """

        if not profile:
            return ""

        if "file" in profile.lower():
            message = (
                f"{greeting} and welcome!\n"
                "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
                "I can answer questions about the contents of the files you upload. To get started:\n\n"
                "  1. Upload one or more documents\n"
                "  2. Ask questions about the files and I will try to answer as best as I can\n\n"
                "Supported file types: Word Documents, PDFs, Text files, Excel files, JSON, and YAML files.\n"
                "Looking forward to our conversation!"
            )
        elif "agent" in profile.lower():
            message = (
                f"{greeting} and welcome!\n"
                "I am Sage, your AI assistant, here to help you orchestrate AI agents using the CrewAI framework.\n\n"
                "CrewAI empowers agents to work together seamlessly, tackling complex tasks through collaborative intelligence.\n"
                "**Note**: Each crew behaves based on its configuration, and responses may take some time.\n\n"
                "To get started, choose a crew from the list below. Then, send your message to the agents and wait for them to kickstart their tasks."
            )
        else:
            message = (
                f"{greeting} and welcome!\n"
                "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
                "I can provide you with data and updates from a variety of sources including:\n"
                f"  {source_repr}\n\n"
                "To get started, simply select an option below; then begin typing your query or ask for help to see what I can do."
            )
        return message.strip()

    def encode_image(self, image_path: str) -> str:
        """
        Encodes an image file into a base64 string.

        This function reads an image file from the provided file path,
        encodes its binary data into a base64 format, and returns the
        encoded string.

        Args:
            image_path (str): The path to the image file to be encoded.

        Returns:
            str: The base64 encoded string representation of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def create_chat_prompt(
        self, system_prompt, user_prompt, additional_user_prompts: list = None
    ):
        """
        Creates a structured chat prompt template for the chat system.

        This method returns a `ChatPromptTemplate` object by combining the
        provided system-level prompt and user-level prompt messages. Additionally,
        it can incorporate a list of extra user prompts, such as images or other media.

        Args:
            system_prompt (str): The primary prompt intended for the system's context.
            user_prompt (str): The main prompt intended for the user's input.
            additional_user_prompts (list, optional): A list of additional user prompts. Each entry
                                                      in the list should be a dictionary specifying
                                                      the type and content of the prompt.

        Returns:
            ChatPromptTemplate: A template object containing the fully structured prompt,
                                ready to be used in the chat system.
        """
        user_messages = [{"type": "text", "text": user_prompt}]
        if additional_user_prompts:
            user_messages.extend(additional_user_prompts)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_messages),
            ]
        )

    def _create_prompt_with_images(
        self, x: dict, system_prompt: str, user_prompt: str
    ) -> ChatPromptTemplate:
        """
        Helper function to create a prompt template based on the presence of image data.

        Args:
            x (dict): A dictionary that may contain image data with keys as follows:
                    - "image_data": A list of dictionaries, each containing:
                        - "mime": The MIME type of the image (e.g., 'image/jpeg').
                        - "path": The file path to the image to be encoded.
            system_prompt (str): The system prompt template.
            user_prompt (str): The user prompt template.

        Returns:
            ChatPromptTemplate: The created prompt template.
        """
        images = x.get("image_data")

        if not images:
            # Standard Prompt Template
            return self.create_chat_prompt(system_prompt, user_prompt)

        images_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.mime};base64,{self.encode_image(image.path)}"
                },
            }
            for image in images
        ]

        # Prompt Template for Multi-Modality (Includes Image Data)
        return self.create_chat_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            additional_user_prompts=images_content,
        )

    def query_generator_complete_prompt(self, x: dict) -> ChatPromptTemplate:
        """
        Routes to the appropriate prompt template based on the presence of image data for the query generator.

        Args:
            x (dict): A dictionary that may contain image data with keys as follows:
                    - "image_data": A list of dictionaries, each containing:
                        - "mime": The MIME type of the image (e.g., 'image/jpeg').
                        - "path": The file path to the image to be encoded.
        """
        return self._create_prompt_with_images(
            x, self.query_generator_system_prompt, self.query_generator_user_prompt
        )

    def qa_complete_prompt(self, x: dict) -> ChatPromptTemplate:
        """
        Routes to the appropriate QA prompt template based on the presence of image data.

        Args:
            x (dict): A dictionary that may contain image data with keys as follows:
                    - "image_data": A list of dictionaries, each containing:
                        - "mime": The MIME type of the image (e.g., 'image/jpeg').
                        - "path": The file path to the image to be encoded.
        """
        return self._create_prompt_with_images(
            x, self.qa_system_prompt, self.qa_user_prompt
        )
