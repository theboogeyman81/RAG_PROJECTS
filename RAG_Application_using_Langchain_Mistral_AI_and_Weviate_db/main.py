from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


WEAVIATE_URL = ""
WEAVIATE_API_KEY = ""


loader=PyPDFLoader("data/sample.pdf",extract_images=True)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

client = weaviate.Client(
    url = WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
)


embedding_model_name = "sentence-transformers/all-mpnet-L6-v2"
embeddings=HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

documents = text_splitter.split_documents(pages)


vectorstore = Weaviate.from_documents(
    documents,
    embeddings,
    client=client,
    by_text=False,
)





