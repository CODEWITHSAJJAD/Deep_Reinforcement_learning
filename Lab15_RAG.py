import getpass
import os
openai_api_key='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'
os.environ["OPENAI_API_KEY"]=getpass.getpass(openai_api_key)
os.environ['USER_AGENT'] ='myagent'
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2017-06-21-overview/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
##################################
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
##################################
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("What is Convolutional Neural Networks?")
vectorstore.delete_collection()
###########################################################################
from langchain_community.document_loaders import PyPDFLoader
# Loading our data
file_path = 'E:\\Biit\\ANN&DL\\ANNDL Lectures-1\\NN-WK-11-Lec-21-22-Transformers.pdf'
loader = PyPDFLoader(file_path)
documents = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# Splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)
# Storing our data into vector database
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.invoke("What is a masked attention")