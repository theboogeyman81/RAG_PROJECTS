import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.schema.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable in your .env file")

with open("state_of_the_union.txt", "r",encoding="utf-8") as f:
    datat=f.read()

loader=TextLoader("state_of_the_union.txt",encoding="utf-8")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=text_splitter.split_documents(documents)

embeddings=GoogleGenerativeAI(api_key=GEMINI_API_KEY)

vectorstore=FAISS.from_documents(chunks,embeddings)

retriever=vectorstore.as_retriever()

template="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt=ChatPromptTemplate.from_template(template)

output_parser=StrOutputParser()

llm=ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY,model="gemini-1.5-flash")

rag_chain=(
    {"context":retriever,"question":RunnablePassthrough()}
    |prompt
    |llm
    |output_parser
)

rag_chain.invoke("How is the United States supporting Ukraine economically and militarily?")
rag_chain.invoke("What action is the U.S. taking to address rising gas prices?")