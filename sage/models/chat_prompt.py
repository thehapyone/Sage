from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate, PromptTemplate

from sage.sources.utils import get_time_of_day_greeting
from sage.utils.sources import Source


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

    qa_template_chat: str = """
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
    
    <context>
    {context}
    </context>
    
    Here is the current chat history - use if relevant:
    <chat_history>
    {chat_history}
    <chat_history/>

    Question: {question}

    REMEMBER: No in-line citations and no citation repetition. State sources in the 'Footnotes' section. For standard conversation and questions about Sage's nature, no footnotes are required. Include footnotes only when they are directly relevant to the provided answer.
    
    Footnotes:
    [1] - Brief summary of the first source. (Less than 10 words)
    [2] - Brief summary of the second source.
    ...continue for additional sources, only if relevant and necessary.  
    """

    # The prompt template for the condense question chain
    condense_prompt = PromptTemplate.from_template(condensed_template)

    qa_prompt = ChatPromptTemplate.from_template(qa_template_chat)
    """The prompt template for the chat complete chain"""

    @property
    def tool_description(self) -> str:
        """Generate a description for the source qa tool"""
        source_description = Source().sources_to_string().replace("\n  ", " ")
        description = f"{self.description_pre}\n. I have access to the following sources: {source_description}"
        return description

    @staticmethod
    def generate_welcome_message(profile: str = "chat"):
        """Generate and format an introduction message."""
        greeting = get_time_of_day_greeting()
        sources = Source().sources_to_string()

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
                f"  {sources}\n\n"
                "To get started, simply select an option below; then begin typing your query or ask for help to see what I can do."
            )
        return message.strip()
