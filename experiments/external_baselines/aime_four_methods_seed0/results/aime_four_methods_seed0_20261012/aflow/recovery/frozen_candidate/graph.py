from typing import Literal
import workspace.rpas_aime_native.MATH.workflows.template.operator as operator
import workspace.rpas_aime_native.MATH.workflows.round_9.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with self-correction
        """
        # Step 1: Generate a detailed derivation and potential answer
        derivation = await self.custom(input=problem, instruction=prompt_custom.DERIVATION_PROMPT)
        
        # Step 2: Use Programmer to extract the integer answer from the derivation
        extraction_result = await self.programmer(problem=problem, analysis=derivation['response'])
        
        # Step 3: If extraction fails or is empty, or if the answer seems invalid, trigger self-correction
        if not extraction_result['output'] or not extraction_result['output'].isdigit():
            # Step 3a: Generate a specific self-correction prompt asking the model to re-evaluate
            correction_prompt = f"""
            You previously attempted to solve: {problem}
            Your previous derivation was: {derivation['response']}
            The system failed to extract a valid integer answer from your response.
            
            Task:
            1. Re-analyze the problem carefully.
            2. Provide a concise derivation.
            3. Output the final integer answer STRICTLY in the format: \boxed{integer}.
            Do not include any other text after the boxed answer.
            """
            correction_response = await self.custom(input=correction_prompt, instruction=prompt_custom.DERIVATION_PROMPT)
            
            # Step 3b: Attempt extraction again on the corrected response
            correction_extraction = await self.programmer(problem=problem, analysis=correction_response['response'])
            
            # Step 3c: If still failing, use ScEnsemble on the original and corrected derivations
            if not correction_extraction['output'] or not correction_extraction['output'].isdigit():
                solutions = [derivation['response'], correction_response['response']]
                final_result = await self.sc_ensemble(solutions=solutions, problem=problem)
                return final_result['response'], self.llm.get_usage_summary()["total_cost"]
            
            return correction_extraction['output'], self.llm.get_usage_summary()["total_cost"]
        
        return extraction_result['output'], self.llm.get_usage_summary()["total_cost"]
