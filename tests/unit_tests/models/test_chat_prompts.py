import base64
from unittest.mock import mock_open, patch

import pytest
from langchain.prompts import ChatPromptTemplate

from sage.models.chat_prompt import ChatPrompt


class MockImageObject:
    def __init__(self, mime, path):
        self.mime = mime
        self.path = path


def test_tool_description():
    chat_prompt = ChatPrompt()
    source_repr = "Source 1: Database"
    expected_description = (
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
        "\n. I have access to the following sources: Source 1: Database"
    )
    assert chat_prompt.tool_description(source_repr) == expected_description


def test_generate_welcome_message_chat():
    chat_prompt = ChatPrompt()
    greeting = "Good morning"
    source_repr = "Source 1: Database\nSource 2: API"
    profile = "chat"
    expected_message = (
        "Good morning and welcome!\n"
        "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
        "I can provide you with data and updates from a variety of sources including:\n"
        "  Source 1: Database\nSource 2: API\n\n"
        "To get started, simply select an option below; then begin typing your query or ask for help to see what I can do."
    )
    assert (
        chat_prompt.generate_welcome_message(greeting, source_repr, profile)
        == expected_message
    )


def test_generate_welcome_message_file():
    chat_prompt = ChatPrompt()
    greeting = "Good morning"
    source_repr = "Source 1: Database\nSource 2: API"
    profile = "file"
    expected_message = (
        "Good morning and welcome!\n"
        "I am Sage, your AI assistant, here to support you with information and insights. How may I assist you today?\n\n"
        "I can answer questions about the contents of the files you upload. To get started:\n\n"
        "  1. Upload one or more documents\n"
        "  2. Ask questions about the files and I will try to answer as best as I can\n\n"
        "Supported file types: Word Documents, PDFs, Text files, Excel files, JSON, and YAML files.\n"
        "Looking forward to our conversation!"
    )
    assert (
        chat_prompt.generate_welcome_message(greeting, source_repr, profile)
        == expected_message
    )


def test_generate_welcome_message_agent():
    chat_prompt = ChatPrompt()
    greeting = "Good morning"
    source_repr = "Source 1: Database\nSource 2: API"
    profile = "agent"
    expected_message = (
        "Good morning and welcome!\n"
        "I am Sage, your AI assistant, here to help you orchestrate AI agents using the CrewAI framework.\n\n"
        "CrewAI empowers agents to work together seamlessly, tackling complex tasks through collaborative intelligence.\n"
        "**Note**: Each crew behaves based on its configuration, and responses may take some time.\n\n"
        "To get started, choose a crew from the list below. Then, send your message to the agents and wait for them to kickstart their tasks."
    )
    assert (
        chat_prompt.generate_welcome_message(greeting, source_repr, profile)
        == expected_message
    )


def test_encode_image():
    chat_prompt = ChatPrompt()
    image_path = "test_image.png"
    with patch("builtins.open", mock_open(read_data=b"test_image_data")):
        expected_encoded_image = base64.b64encode(b"test_image_data").decode()
        assert chat_prompt.encode_image(image_path) == expected_encoded_image


def test_create_chat_prompt():
    chat_prompt = ChatPrompt()
    system_prompt = "System prompt"
    user_prompt = "User prompt"
    additional_user_prompts = [{"type": "image_url", "image_url": {"url": "test_url"}}]
    expected_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": "test_url"}},
                ],
            ),
        ]
    )
    assert (
        chat_prompt.create_chat_prompt(
            system_prompt, user_prompt, additional_user_prompts
        )
        == expected_prompt
    )


def test_query_generator_complete_prompt():
    chat_prompt = ChatPrompt()
    x = {"image_data": [MockImageObject(mime="image/jpeg", path="test_image.jpg")]}
    with patch.object(chat_prompt, "encode_image", return_value="encoded_image_data"):
        expected_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", chat_prompt.query_generator_system_prompt),
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": chat_prompt.query_generator_user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,encoded_image_data"
                            },
                        },
                    ],
                ),
            ]
        )
        assert chat_prompt.query_generator_complete_prompt(x) == expected_prompt


def test_qa_complete_prompt():
    chat_prompt = ChatPrompt()
    x = {"image_data": [MockImageObject(mime="image/jpeg", path="test_image.jpg")]}
    with patch.object(chat_prompt, "encode_image", return_value="encoded_image_data"):
        expected_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", chat_prompt.qa_system_prompt),
                (
                    "user",
                    [
                        {"type": "text", "text": chat_prompt.qa_user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,encoded_image_data"
                            },
                        },
                    ],
                ),
            ]
        )
        assert chat_prompt.qa_complete_prompt(x) == expected_prompt
