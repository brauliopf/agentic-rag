import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_URL = os.getenv("PINECONE_HOST_URL")
PINECONE_INDEX_NAME = "agentic-rag"  # Choose a name for your index

# Model configurations
EMBEDDING_MODEL = "text-embedding-3-small"  # Cost-effective, good performance
LLM_MODEL = "gpt-3.5-turbo"  # Good balance of quality and cost

# Vector dimensions for OpenAI embedding models
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small