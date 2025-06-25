from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


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


#print(vectorstore.similarity_search(
#    query_texts=["What is the main idea of the document?"],
#    k=3
#))


template = """You are an assistant for question-answering tasks.
Use the following pieces of retrived context to answer the question.
If you don't know the answer,just say that you don't know.
Use ten sentences maxium and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt= ChatPromptTemplate.from_template(template)

#from google.colab import userdata
#huggingfacehub_api_token = userdata.get('Huggingface_token')


model = HuggingFaceHub(
    huggingfacehub_api_token = huggingfacehub_api_token,
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.1',
    model_kwargs={"temperature":1,"max_length":180}
)


output = StrOutputParser()

retriver = vectorstore.as_retriever()

rag_chain= (
    {"context":retriver,"question":RunnablePassthrough()}
    |prompt
    |model
    |output
)


rag_chain.invoke()













