import rag_core

def main():
    print("Initializing Calculus AI Tutor (CLI Version)...")
    
    # 1. Load Data
    chunks = rag_core.load_knowledge_base()
    if not chunks:
        print("Knowledge base not found. Building it now (this may take a while)...")
        try:
            # Simple progress printer
            def progress(p, text):
                if p % 10 == 0:
                    print(f"{p}% - {text}")
            
            chunks = rag_core.build_knowledge_base(progress)
        except Exception as e:
            print(f"Error building knowledge base: {e}")
            return

    print(f"Knowledge base ready with {len(chunks)} chunks.")
    print("\n--- You can now chat with the Calculus Tutor. Type 'exit' to quit. ---\n")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if not user_input.strip():
            continue

        print("Searching textbook...")
        try:
            context = rag_core.retrieve_context(user_input, chunks)
            
            print("Thinking...")
            answer = rag_core.generate_answer(user_input, context)
            print(f"\nAI: {answer}")
        
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
