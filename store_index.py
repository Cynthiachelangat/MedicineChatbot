from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve Pinecone API key from the .env file
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Load and process the PDF
extracted_data = load_pdf("data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Create an instance of Pinecone with the API key
pinecone_client = Pinecone(api_key=pinecone_api_key)

# Name of the index
index_name = "medical-chatbot"

# List existing indexes and check if the index exists
existing_indexes = pinecone_client.list_indexes()

# Check if the index exists, and create it if it does not
if index_name not in existing_indexes:
    pinecone_client.create_index(
        name=index_name,
        dimension=384,  # Adjust dimension based on your embedding model
        metric="cosine",  # Adjust the metric if necessary
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Specify cloud provider and region
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the existing index
index = pinecone_client.Index(index_name)

# Now proceed with creating embeddings for the text chunks
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)

print(f"Connected to Pinecone index: {index_name}")
