from pathlib import Path

from langchain.llms import OpenAI
import chromadb

from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
#from langchain_community.embeddings import GPT4AllEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("loading model")
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

fulldir = Path.home() / 'OneDrive' / 'Documents' / 'throawaylien'
# C:\Users\joshs\OneDrive\Documents\throawaylien
#loaderTEXT = TextLoader(pathy)
dirloader = DirectoryLoader(fulldir.absolute(), glob='**/*.txt', loader_cls=TextLoader)
#loaderPDF = PyPDFLoader(pathypdf)
print("instantiated loader")
dirdata = dirloader.load()

# print("Data was: ", data)
print("splitting text and embedding using gpt4all embeddings")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
splits = text_splitter.split_documents(dirdata)

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    api_key="n/a",
    model="nomic-ai/nomic-embed-text-v1.5-GGUF",
    # model="text-embedding-3-small",
    # embedding_ctx_length=1000,
    # tiktoken_enabled=True,
    )
new_client = chromadb.EphemeralClient()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
print("finished the vectorestore")
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = client

template = """Use the provided pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

CONTEXT:

```{context}```

QUESTION: {question}

HELPFUL ANSWER:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def enter_question():
    print("about to invoke the rag_chain")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    question = input("Enter your prompt: ")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("just finished invoking the rag_chain")
    # cleanup

while True:
    enter_question()

vectorstore.delete_collection()
