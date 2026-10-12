DERIVATION_PROMPT = """
You are an expert mathematician solving a competition math problem.
Your task is to provide a step-by-step derivation of the solution.
At the very end of your response, you must output the final integer answer in the exact format: \boxed{integer}.
Do not include any other text after the boxed answer.
Ensure the integer is valid (e.g., positive if implied by context).

Problem: {input}
"""