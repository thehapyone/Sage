class MockSession:
    def __init__(self):
        self.user_session = {}

    def get(self, key, default=None):
        return self.user_session.get(key, default)

    def set(self, key, value):
        self.user_session[key] = value
