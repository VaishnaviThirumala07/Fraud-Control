import os
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def initialize_rag(knowledge_base_dir: str = "knowledge_base"):
    """
    Initializes the RAG system by loading documents from the specified directory
    and creating a vector store index.
    """
    print(f"Loading documents from {knowledge_base_dir}...")
    
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding()

    # 2. Load Documents
    if not os.path.exists(knowledge_base_dir):
        raise FileNotFoundError(f"Directory '{knowledge_base_dir}' not found.")
        
    documents = SimpleDirectoryReader(knowledge_base_dir).load_data()
    print(f"Loaded {len(documents)} documents.")

    # 3. Create the Index
    print("Building Vector Store Index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 4. Create the Query Engine
    query_engine = index.as_query_engine(similarity_top_k=3)
    return query_engine

def query_rag(query_engine, user_query: str):
    """
    Queries the RAG system and returns the response.
    """
    print(f"\nQuerying: {user_query}")
    response = query_engine.query(user_query)
    
    print("\n--- Response ---")
    print(response)
    print("\n--- Sources ---")
    for source in response.source_nodes:
        print(f"- {source.node.metadata.get('file_name', 'Unknown')}: Score {source.score:.3f}")
        
    return response

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Please add it to your .env file.")
    else:
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        
        engine = initialize_rag(knowledge_base_dir=kb_path)
        query_rag(engine, "What is Trade-Based Money Laundering?")
        query_rag(engine, "What is Enhanced Due Diligence (EDD) and when is it applied?")
