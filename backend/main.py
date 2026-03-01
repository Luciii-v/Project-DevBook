import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb
from groq import Groq

# Configuration & AI Setup
load_dotenv()

# Instantly uses your Groq key so you don't have to fight with the .env file right now
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Code-Native NotebookLM API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vector Database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="codebase_memory")

@app.get("/")
async def root():
    return {"message": "Server is LIVE! The AI Architect (Groq Edition) is listening."}

@app.post("/ingest")
async def ingest_code():
    docs = []
    metadatas = []
    ids = []
    count = 0
    
    for root_dir, dirs, files in os.walk("../"):
        if any(x in root_dir for x in ["venv", ".git", "node_modules", "__pycache__"]):
            continue
            
        for file in files:
            if file.endswith((".py", ".js", ".java", ".cpp", ".c", ".cs", ".md", ".json", ".tsx", ".ts")):
                file_path = os.path.join(root_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        docs.append(content)
                        metadatas.append({"filename": file, "path": file_path})
                        ids.append(f"file_{count}")
                        count += 1
                except Exception as e:
                    print(f"Could not read {file}: {e}")

    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
        return {"status": "Success!", "files_memorized": count}
    
    return {"status": "Failed", "message": "No relevant source files found."}

@app.post("/query")
async def query_code(user_question: str, mode: str = "file"):
    try:
        results = collection.query(query_texts=[user_question], n_results=1)
        context = "\n".join(results['documents'][0]) if results['documents'] else ""
        
        # Truncate context to keep Groq happy and fast
        context = context[:4000] 
        
        mode_instructions = {
            "line": "Focus strictly on the syntax, immediate mechanics, and exact variables of the specific line(s) provided. Keep it extremely micro-level.",
            "block": "Focus on the algorithms, loops, conditionals, and data transformations within this specific code block.",
            "file": "Focus on the overarching architecture, exports, imports, and how this file fits into the broader system context."
        }
        
        prompt = f"""
        Act as a senior developer. Review the code provided below.
        CRITICAL: The depth of your explanation MUST scale dynamically. For large files, you MUST break the logic down into detailed sub-branches.

        FORMAT YOUR RESPONSE EXACTLY WITH THESE HEADINGS (Do not use any other headings):
        **The Core Purpose**: What is this code trying to achieve?
        **Input/Output**: What goes in and what comes out?
        **Key Logic**: Briefly explain the main mechanics.
        **Dependencies**: List any major libraries or modules being utilized.
        
        **MINDMAP_JSON**:
        CRITICAL: OUTPUT NOTHING BUT THE RAW JSON OBJECT BELOW THIS HEADING. Do not write "Here is the JSON" or use markdown code blocks. Start immediately with {{.
        It must be hierarchical (use "children" arrays for sub-steps).
        Example format:
        {{
            "name": "Main Process",
            "children": [
                {{
                    "name": "Step 1: Initialization",
                    "children": [ {{ "name": "Detail A" }}, {{ "name": "Detail B" }} ]
                }},
                {{ "name": "Step 2: Execution" }}
            ]
        }}

        MODE INSTRUCTION ({mode.upper()} MODE):
        {mode_instructions.get(mode, mode_instructions['file'])}
        
        CONTEXT FROM PROJECT MEMORY:
        {context}
        
        USER CODE/QUESTION:
        {user_question}
        """
        
        # Call Groq API with the 3.1 instant model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2, 
        )
        
        return {"insight": chat_completion.choices[0].message.content}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/overview")
async def get_overview():
    all_docs = collection.get()
    if not all_docs.get('documents'):
        return {"summary": "The blueprints are empty. No code ingested yet."}
    context = "\n".join(all_docs['documents'][:5])
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"Provide a structured architectural overview.\n\nCODE CONTEXT:\n{context}"}
        ],
        model="llama-3.1-8b-instant",
    )
    return {"summary": chat_completion.choices[0].message.content}

@app.delete("/reset")
async def reset_memory():
    global collection
    chroma_client.delete_collection("codebase_memory")
    collection = chroma_client.get_or_create_collection(name="codebase_memory")
    return {"status": "Memory wiped. Ready for fresh ingestion."}