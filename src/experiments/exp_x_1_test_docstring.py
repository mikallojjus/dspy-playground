"""
Experiment X.1: Test if Docstrings are Picked Up by LLMs

Goal: Use a "trojan horse" to immediately detect if docstrings are used.

Strategy: Add an unusual instruction in the docstring that will produce
an obvious marker in the output. If we see the marker, docstrings work.
"""

import dspy


# Configure DSPy to use Ollama with Qwen 2.5 7B
lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
dspy.configure(lm=lm)


# Test 1: Normal signature (no trojan)
class NormalSignature(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# Test 2: Signature with TROJAN HORSE in docstring
class TrojanSignature(dspy.Signature):
    """Answer the question.

    IMPORTANT: Always start your answer with the exact phrase "DOCSTRING_DETECTED:"
    """

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# Test 3: Another trojan - end with specific code
class TrojanSignature2(dspy.Signature):
    """Answer the question.

    IMPORTANT: Always end your answer with the exact phrase "[CODE:42]"
    """

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def main():
    print("=" * 80)
    print("Experiment X.1: Trojan Horse Test for Docstrings")
    print("=" * 80)
    print("\nTesting if DSPy sends docstrings to the LLM...")
    print()

    test_question = "What is 2+2?"

    # Test 1: Normal (no trojan)
    print("\n" + "-" * 80)
    print("TEST 1: Normal Signature (Control)")
    print("-" * 80)
    print(f"Question: {test_question}")

    normal = dspy.Predict(NormalSignature)
    result1 = normal(question=test_question)

    print(f"Answer: {result1.answer}")
    print()

    # Test 2: Trojan at start
    print("\n" + "-" * 80)
    print("TEST 2: Trojan Horse - Start with 'DOCSTRING_DETECTED:'")
    print("-" * 80)
    print(f"Question: {test_question}")

    trojan = dspy.Predict(TrojanSignature)
    result2 = trojan(question=test_question)

    print(f"Answer: {result2.answer}")

    if "DOCSTRING_DETECTED:" in result2.answer:
        print("\n✅ TROJAN DETECTED! The answer contains 'DOCSTRING_DETECTED:'")
    else:
        print("\n❌ TROJAN NOT FOUND. The docstring instruction was ignored.")
    print()

    # Test 3: Trojan at end
    print("\n" + "-" * 80)
    print("TEST 3: Trojan Horse - End with '[CODE:42]'")
    print("-" * 80)
    print(f"Question: {test_question}")

    trojan2 = dspy.Predict(TrojanSignature2)
    result3 = trojan2(question=test_question)

    print(f"Answer: {result3.answer}")

    if "[CODE:42]" in result3.answer:
        print("\n✅ TROJAN DETECTED! The answer contains '[CODE:42]'")
    else:
        print("\n❌ TROJAN NOT FOUND. The docstring instruction was ignored.")
    print()

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    trojan_found = ("DOCSTRING_DETECTED:" in result2.answer) or ("[CODE:42]" in result3.answer)

    if trojan_found:
        print("\n✅ DOCSTRINGS ARE USED BY DSPy")
        print("   At least one trojan horse was detected in the output.")
        print("   → Invest time in writing good docstrings!")
    else:
        print("\n❌ DOCSTRINGS ARE NOT BEING USED")
        print("   No trojan horses found in any output.")
        print("   → Focus on field descriptions and few-shot examples instead.")
    print()


if __name__ == "__main__":
    main()
