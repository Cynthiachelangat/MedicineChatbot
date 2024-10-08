from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_api_key = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("../data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone with the API key
pinecone_client = Pinecone(api_key=pinecone_api_key)

# Name of the index
index_name = "medical-chatbot"
# Connect to the existing index
index = pinecone_client.Index(index_name)

# Now proceed with creating embeddings for the text chunks
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)