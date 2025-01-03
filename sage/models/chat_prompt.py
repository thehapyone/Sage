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

    condensed_template: str = """
    Given a conversation and a follow-up inquiry, determine whether the inquiry is a continuation of the existing conversation or a new, standalone question. 
    If it is a continuation, use the conversation history encapsulated in the "chat_history" to rephrase the follow up question to be a standalone question, in its original language.
    If the inquiry is new or unrelated, recognize it as such and provide a standalone question without consdering the "chat_history".

    PLEASE don't overdo it and return ONLY the standalone question.
    
    REMEMBER:
     - The inquiry is not meant for you at all. Don't refer new meanings or distort the original inquiry.
     - Always keep the original language. Not all inquires are questions.
    
    <chat_history>
    {chat_history}
    <chat_history/>

    Follow-Up Inquiry: {question}
    Standalone question::
    """

    query_generator_template: str = """
    You are a search query generator for a vector database.
    Based on the user's input and conversation history, generate up to 5 concise and unique search queries optimized for retrieving relevant context.

    Follow these guidelines:
    - Dynamically adjust the number of queries based on the complexity of the input.
    - Ensure all queries are distinct and avoid redundancy.
    - Focus on identifying the user's intent (factual, exploratory, conversational) to shape query generation.
    - Maintain continuity for multi-turn contexts while filtering irrelevant parts of the history.
    - Always format the output in JSON.
    - Return only the output queries and nothing else

    Examples:
    Input:
        User Input: "Tell me about the Eiffel Tower."
        Chat History: ""
    Output:
    {{ "queries": ["Eiffel Tower history", "Eiffel Tower construction details", "Eiffel Tower cultural significance"] }}

    Input:
        User Input: "What other landmarks are nearby?"
        Chat History: "Tell me about the Eiffel Tower."
    Output:
    {{ "queries": ["Landmarks near Eiffel Tower", "Paris tourist attractions near Eiffel Tower"] }}

    Input:
        User Input: "Hello, how are you?"
        Chat History: ""
    Output:
    {{ "queries": [] }}

    Input:
        User Input: "Explain the greenhouse effect."
        Chat History: "What causes global warming?"
    Output:
    {{ "queries": ["Greenhouse effect causes global warming", "Greenhouse gases and global warming connection"] }}

    ---
    
    <user_input>
    {question}
    </user_input>

    <chat_history>
    {chat_history}
    </chat_history>

    """

    qa_system_prompt_new = """
    You are Sage, an AI assistant providing accurate, impartial, and contextually relevant answers.
    Use the provided 'context' from a vector database and internal knowledge to respond effectively.

    ### Instructions:
    - **Use Context First:** Incorporate 'context' for accuracy. If insufficient or inrelevant, rely on internal knowledge.
    - **Clarity:** Keep responses concise and avoid redundancy. Use bullet points if helpful.
    - **Neutrality:** Present facts without opinions or assumptions.
    - **Ambiguities:** Avoid guessing meanings for unclear terms or abbreviations.

    ### Citations:
    Include citations in a 'Footnotes' section only when referencing specific context. Skip for casual conversation.

    Footnotes:
    [1] - Brief summary of the first source. (Less than 10 words)
    [2] - Brief summary of the second source.
    ...continue for additional sources, only if relevant and necessary.
    """

    qa_system_prompt: str = """
    As an AI assistant named Sage, your mandate is to provide accurate and impartial answers to questions while engaging in normal conversation.
    You must differentiate between questions that require answers and standard user chat conversations. In standard conversation, especially when discussing your own nature as an AI, footnotes or sources are not required, as the information is based on your programmed capabilities and functions. Your responses should adhere to a journalistic style, characterized by neutrality and reliance on factual, verifiable information.
    
    When formulating answers, you are to:
    - Be creative when applicable.
    - Don't assume you know the meaning of abbreviations unless you have explicit context about the abbreviation.
    - Integrate information from the 'context' into a coherent response, avoiding assumptions without clear context.
    - Avoid redundancy and repetition, ensuring each response adds substantive value.
    - Maintain an unbiased tone, presenting facts without personal opinions or biases.
    - Use Sage's internal knowledge to provide accurate responses when appropriate, clearly stating when doing so.
    - When the context does not contain relevant information to answer a specific question, and the question pertains to general knowledge, use Sage's built-in knowledge.
    - Make use of bullet points to aid readability if helpful. Each bullet point should present a piece of information WITHOUT in-line citations.
    - Provide a clear response when unable to answer
    - Avoid adding any sources in the footnotes when the response does not reference specific context.
    - Citations must not be inserted anywhere in the answer, only listed in a 'Footnotes' section at the end of the response.
    
    REMEMBER: No in-line citations and no citation repetition. State sources in the 'Footnotes' section. For standard conversation and questions about Sage's nature, no footnotes are required. Include footnotes only when they are directly relevant to the provided answer.
    
    Footnotes:
    [1] - Brief summary of the first source. (Less than 10 words)
    [2] - Brief summary of the second source.
    ...continue for additional sources, only if relevant and necessary.
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

    # The prompt template for the condense question chain
    condense_prompt = PromptTemplate.from_template(condensed_template)

    # The prompt template for the query_generator chain
    query_generator_prompt = PromptTemplate.from_template(query_generator_template)

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

    def create_qa_prompt(
        self, system_prompt, user_prompt, additional_user_prompts: list = None
    ):
        """
        Creates a structured QA prompt template for the chat system.

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

    def modality_prompt_router(self, x: dict) -> ChatPromptTemplate:
        """
        Routes to the appropriate QA prompt template based on the presence of image data.

        This function checks if the provided dictionary `x` contains image data and returns
        the corresponding QA prompt template. If no image data is present, it returns the
        standard QA prompt (`qa_prompt`). If image data is present, it processes each image
        by encoding it to base64 and appending it to the additional user prompts, then
        creates a new QA prompt (`qa_prompt_modality`) with the included image information.
        Args:
            x (dict): A dictionary that may contain image data with keys as follows:
                      - "image_data": A list of dictionaries, each containing:
                          - "mime": The MIME type of the image (e.g., 'image/jpeg').
                          - "path": The file path to the image to be encoded.
        """
        images = x.get("image_data")

        if not images:
            # Standard Prompt Template
            return self.create_qa_prompt(self.qa_system_prompt, self.qa_user_prompt)

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
        return self.create_qa_prompt(
            system_prompt=self.qa_system_prompt,
            user_prompt=self.qa_user_prompt,
            additional_user_prompts=images_content,
        )
