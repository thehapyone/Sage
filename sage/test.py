from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory( k=3, ai_prefix="Assistant")
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

print(memory.load_memory_variables({}).get("history"))
