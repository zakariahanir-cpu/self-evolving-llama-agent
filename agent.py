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
        # تم تعديل التنسيق هنا ليكون أوضح في ملف الذاكرة
        self.memory += "\n- " + new_experience.strip()
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(self.memory)
        print(f"Memory updated and saved to {self.memory_file}")

    def run_task(self, task_prompt):
        print(f"\nAgent received task: {task_prompt}")
        prompt = f"""You are a self-evolving AI agent. Your current memory: {self.memory} 
Your goal is to complete the task and then reflect on what you learned to improve your future performance. 
You MUST follow this format exactly: 
Agent's Response: [Your answer here] 
Reflection: [Your thoughts on the process] 
New Experience: [A concise lesson or fact to add to your memory] 

Task: {task_prompt} """

        # تم تقليل الكلمات التوقف (stop) لضمان أن النموذج يكمل كتابة التجربة الجديدة
        output = self.llm(prompt, max_tokens=512, stop=["Task:"], echo=False)
        response_text = output["choices"][0]["text"].strip()
        print(f"\nAgent's Raw Output:\n{response_text}")

        # --- بداية الإصلاح الجذري لمشكلة الحفظ ---
        agent_response = ""
        reflection = ""
        new_experience = ""

        # تحويل النص لصغير للبحث فقط، لكن الاستخراج يتم من النص الأصلي
        text_lower = response_text.lower()
        ref_marker = "reflection:"
        exp_marker = "new experience:"

        ref_pos = text_lower.find(ref_marker)
        exp_pos = text_lower.find(exp_marker)

        if ref_pos != -1:
            agent_response = response_text[:ref_pos].strip()
            if exp_pos != -1 and exp_pos > ref_pos:
                reflection = response_text[ref_pos + len(ref_marker):exp_pos].strip()
                new_experience = response_text[exp_pos + len(exp_marker):].strip()
            else:
                reflection = response_text[ref_pos + len(ref_marker):].strip()
        else:
            agent_response = response_text.strip()
        # --- نهاية الإصلاح ---

        print(f"\n--- Agent's Response ---\n{agent_response}")
        print(f"\n--- Agent's Reflection ---\n{reflection}")

        if new_experience and new_experience.lower() != "none":
            self._save_memory(new_experience)
        else:
            print("No new experience suggested for memory update.")

        return agent_response

if __name__ == "__main__":
    model_file = os.getenv("LLAMA_MODEL_PATH", "./model.gguf")
    if not os.path.exists(model_file):
        print(f"Error: Llama model file not found at {model_file}")
        exit(1)

    agent = SelfEvolvingAgent(model_file)
    
    # ضع هنا المهام التي تريد تنفيذها
    tasks = ["Summarize the key benefits of using version control systems like Git."]

    for task in tasks:
        agent.run_task(task)

    print("\nAgent has completed all tasks. Check agent_memory.txt for its updated memory.")
    
