import os
from dotenv import load_dotenv
from graph import build_graph
from topic_generator import generate_topic

def main():
    load_dotenv()
    
    # Check if vector store exists
    if not os.path.exists("data/vectorstore"):
        print("Warning: Vector store not found. Creating one if PDFs are present...")
        from ingest import ingest_textbooks
        ingest_textbooks()
    
    app = build_graph()
    
    while True:
        print("\n--- Topic Generation Setup ---")
        career_path = input("Enter your desired Career Path (e.g., 'Computer Scientist', 'Architect'): ")
        curriculum = input("Enter your desired Curriculum/Subject (e.g., 'Linear Algebra', 'Calculus'): ")
        
        if not career_path or not curriculum:
            print("Both fields are required. Please try again.")
            continue
            
        print("Generating optimal research topic...")
        topic = generate_topic(career_path, curriculum)
        
        print(f"\nProposed Topic: {topic}")
        confirm = input("Do you like this topic? (y/n): ").strip().lower()
        
        if confirm == 'y':
            break
        else:
            print("Let's try again with different inputs.")
    
    initial_state = {
        "topic": topic,
        "revision_count": 0,
        "research_data": [], 
        "image_paths": []
    }
    
    print("Starting Research Mate Report AI...")
    final_state = initial_state.copy()
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            final_state.update(value)
            
    # Print Final Report
    report = final_state.get("final_report")
    
    if report:
        with open("report_output.md", "w") as f:
            f.write(report)
    else:
        print("Error: No report generated.")

        
    print("\n\nReport generated successfully: report_output.md")

if __name__ == "__main__":
    main()
