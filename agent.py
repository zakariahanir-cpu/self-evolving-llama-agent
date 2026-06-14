
from llama_cpp import Llama
import os

class SelfEvolvingAgent:
    def __init__(self, model_path, memory_file="agent_memory.txt"):
        self.model_path = model_path
        self.memory_file = memory_file
        self.llm = None
        self._load_model()
        self._load_memory()

    def _load_model(self):
        print(f"Loading Llama model from {self.model_path}...")
        # Assuming a GGUF model file. n_ctx can be adjusted based on model and task.
        self.llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
        print("Llama model loaded.")

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.memory = f.read()
            print(f"Loaded memory from {self.memory_file}")
        else:
            self.memory = "The agent is new and has no prior experiences."
            print("No existing memory found. Starting with a fresh memory.")

    def _save_memory(self, new_experience):
        self.memory += "\n" + new_experience
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(self.memory)
        print(f"Memory updated and saved to {self.memory_file}")

    def run_task(self, task_prompt):
        print(f"\nAgent received task: {task_prompt}")
        prompt = f"""You are a self-evolving AI agent. 
Your current memory:
{self.memory}

Your goal is to complete the task and then reflect on what you learned to improve your future performance.
You MUST follow this format exactly:
Agent's Response: [Your answer here]
Reflection: [Your thoughts on the process]
New Experience: [A concise lesson or fact to add to your memory]

Task: {task_prompt}
"""

        output = self.llm(prompt, max_tokens=512, stop=["Task:", "Agent's Response:"], echo=False)
        response_text = output["choices"][0]["text"].strip()

        print(f"\nAgent's Raw Output:\n{response_text}")

        # Attempt to parse response and reflection
        agent_response = ""
        reflection = ""
        new_experience = ""

        # Simple parsing based on keywords
        parts = response_text.split("Reflection:")
        agent_response = parts[0].strip()

        if len(parts) > 1:
            reflection_parts = parts[1].split("New Experience:")
            reflection = reflection_parts[0].strip()
            if len(reflection_parts) > 1:
                new_experience = reflection_parts[1].strip()

        if not reflection and "Reflection:" in response_text:
            # Fallback if split didn't work as expected due to missing newline
            reflection_start = response_text.find("Reflection:")
            new_experience_start = response_text.find("New Experience:")
            if reflection_start != -1:
                agent_response = response_text[:reflection_start].strip()
                if new_experience_start != -1 and new_experience_start > reflection_start:
                    reflection = response_text[reflection_start + len("Reflection:"):new_experience_start].strip()
                    new_experience = response_text[new_experience_start + len("New Experience:"):].strip()
                else:
                    reflection = response_text[reflection_start + len("Reflection:"):].strip()

        print(f"\n--- Agent's Response ---\n{agent_response}")
        print(f"\n--- Agent's Reflection ---\n{reflection}")

        if new_experience:
            self._save_memory(new_experience)
        else:
            print("No new experience suggested for memory update.")

        return agent_response

if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/model.gguf' with the actual path to your Llama GGUF model file.
    # You will need to download a Llama model in GGUF format (e.g., from Hugging Face).
    # For example: 'Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf'
    # Make sure the model file is accessible from where you run this script.
    model_file = os.getenv("LLAMA_MODEL_PATH", "./model.gguf") # Default to ./model.gguf

    if not os.path.exists(model_file):
        print(f"Error: Llama model file not found at {model_file}")
        print("Please download a Llama GGUF model and place it at this path, or set the LLAMA_MODEL_PATH environment variable.")
        exit(1)

    agent = SelfEvolvingAgent(model_file)

    # Example tasks
    tasks = [
        "Explain the concept of photosynthesis in simple terms.",
        "Write a short Python function to calculate the factorial of a number.",
        "Describe the main challenges of renewable energy adoption.",
        "Propose a simple solution for managing daily tasks effectively.",
        "Summarize the key benefits of using version control systems like Git."
    ]

    for i, task in enumerate(tasks):
        print(f"\n==================== Running Task {i+1}/{len(tasks)} ====================")
        agent.run_task(task)
        print("=================================================================")
        # Add a small delay to allow for processing/saving if needed
        # import time
        # time.sleep(5)

    print("\nAgent has completed all tasks. Check agent_memory.txt for its updated memory.")
