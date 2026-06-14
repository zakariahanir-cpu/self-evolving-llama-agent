from llama_cpp import Llama
import os
import re

class SelfEvolvingAgent:
    def __init__(self, model_path, memory_file="agent_memory.txt"):
        self.model_path = model_path
        self.memory_file = memory_file
        self.llm = None
        self._load_model()
        self._load_memory()

    def _load_model(self):
        print(f"Loading Llama model from {self.model_path}...")
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
        # إضافة التجربة الجديدة مع سطر جديد
        self.memory += "\n- " + new_experience.strip()
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(self.memory)
        print(f"Memory updated and saved to {self.memory_file}")

    def run_task(self, task_prompt):
        print(f"\nAgent received task: {task_prompt}")
        
        prompt = f"""You are a self-evolving AI agent. 
Current Memory: {self.memory}

Task: {task_prompt}

You MUST follow this format exactly:
Agent's Response: [Your answer]
Reflection: [What you learned]
New Experience: [A concise lesson to remember]
"""
        
        output = self.llm(prompt, max_tokens=512, stop=["Task:"], echo=False)
        response_text = output["choices"][0]["text"].strip()
        print(f"\nRaw Output:\n{response_text}")

        # تحسين عملية الاستخراج باستخدام Regex لتكون أكثر مرونة
        new_experience_match = re.search(r"New Experience:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
        
        if new_experience_match:
            new_experience = new_experience_match.group(1).strip()
            if new_experience and new_experience.lower() != "none":
                self._save_memory(new_experience)
            else:
                print("No meaningful new experience suggested.")
        else:
            print("Could not parse 'New Experience' from response.")
            
        return response_text

if __name__ == "__main__":
    model_file = os.getenv("LLAMA_MODEL_PATH", "./model.gguf")
    
    if not os.path.exists(model_file):
        print(f"Error: Model not found at {model_file}")
        exit(1)

    agent = SelfEvolvingAgent(model_file)
    
    # مثال لمهمة
    task = "Summarize the key benefits of using version control systems like Git."
    agent.run_task(task)
    
