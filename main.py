import os
from dotenv import load_dotenv
from graph import build_graph

def main():
    load_dotenv()
    
    # Check if vector store exists
    if not os.path.exists("data/vectorstore"):
        print("Warning: Vector store not found. Creating one if PDFs are present...")
        from ingest import ingest_textbooks
        ingest_textbooks()
    
    app = build_graph()
    
    topic = input("Enter the research topic (e.g., 'Catenary curves in architecture'): ")
    if not topic:
        topic = "Catenary curves in architecture"
    
    initial_state = {
        "topic": topic,
        "revision_count": 0,
        "research_data": [], 
        "image_paths": []
    }
    
    print("Starting Research Mate Report AI...")
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            
    # Print Final Report
    # Ideally, we would retrieve the final state, but stream yields updates.
    # We'll just grab the state at the end if we were running invoke, 
    # but with stream, we need to track it or just trust the file system artifacts if we saved them.
    # Let's run invoke for simplicity in getting the final state return.
    
    final_state = app.invoke(initial_state)
    report = final_state.get("final_report")
    
    with open("report_output.md", "w") as f:
        f.write(report)
        
    print("\n\nReport generated successfully: report_output.md")

if __name__ == "__main__":
    main()
