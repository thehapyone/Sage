from unittest.mock import AsyncMock, MagicMock


class MockSession:
    def __init__(self):
        self.user_session = {}

    def get(self, key, default=None):
        return self.user_session.get(key, default)

    def set(self, key, value):
        self.user_session[key] = value


class MockMessage:
    def __init__(self, *args, **kwargs):
        self.content = kwargs.get("content", "")
        self.send = AsyncMock()
        self.update = AsyncMock()


def create_mock(**kwargs):
    mock_instance = MagicMock()
    mock_instance.configure_mock(**kwargs)
    return mock_instance
