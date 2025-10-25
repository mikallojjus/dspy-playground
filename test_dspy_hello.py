import dspy

# Configure DSPy to use Ollama
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Define simple signature
class BasicQA(dspy.Signature):
    """Answer questions concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Use it
qa = dspy.Predict(BasicQA)
result = qa(question="What is 2+2?")
print(f"Answer: {result.answer}")
