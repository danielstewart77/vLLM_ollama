# import os
# import shlex
# from huggingface_hub import snapshot_download

# MODEL_DIR = "models"

# def pull(model_id: str):
#     sanitized = model_id.lower().replace("/", "-")
#     target_dir = os.path.join(MODEL_DIR, sanitized)

#     if os.path.exists(target_dir):
#         print(f"✔ Model already exists at {target_dir}")
#         return

#     print(f"⬇ Pulling {model_id} to {target_dir}...")
#     snapshot_download(
#         repo_id=model_id,
#         local_dir=target_dir,
#         local_dir_use_symlinks=False,
#         resume_download=True
#     )
#     print("✅ Done!")

# def list_models():
#     if not os.path.exists(MODEL_DIR):
#         print("⚠ No models directory found.")
#         return
#     models = [f for f in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, f))]
#     for model in models:
#         print(f"• {model}")

# def help_menu():
#     print("""
# Available commands:
#   pull <model_id>      Pull a model from HuggingFace
#   list                 List local models
#   help                 Show this help message
#   exit / quit          Exit the CLI
# """)

# def repl():
#     print("🔧 vLLM Model CLI (type 'help' for options)")
#     while True:
#         try:
#             raw = input("> ").strip()
#             if not raw:
#                 continue
#             parts = shlex.split(raw)
#             cmd = parts[0]
#             args = parts[1:]

#             if cmd in ["exit", "quit"]:
#                 break
#             elif cmd == "help":
#                 help_menu()
#             elif cmd == "pull" and args:
#                 pull(args[0])
#             elif cmd == "list":
#                 list_models()
#             else:
#                 print("❌ Unknown command. Type 'help' for a list of commands.")
#         except KeyboardInterrupt:
#             print("\nExiting...")
#             break

# if __name__ == "__main__":
#     repl()
