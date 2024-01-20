from gpt4all import GPT4All
from pathlib import Path
import os
import bs4
#from openai import OpenAI

# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
print("loading model")
# model = GPT4All('C:\\Users\\joshs\\.cache\\lm-studio\\models\\TheBloke\dolphin-2.6-mixtral-8x7b.Q4_K_M.gguf', device='amd') # device='amd', device='intel'
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

#GPT4All(model_name='neuralbeagle14-7b.Q5_K_S.gguf', model_path=Path.home() / '.cache' / 'lm-studio' / 'models' / 'TheBloke' / 'NeuralBeagle14-7B-GGUF', device='cpu', allow_download=False) # device='amd', device='intel'
# model = GPT4All(model_name='mixtral-8x7b-instruct-v0.1.Q5_0.gguf', model_path=Path.home() / '.cache' / 'lm-studio' / 'models' / 'TheBloke' / 'Mixtral-8x7B-Instruct-v0.1-GGUF', device='cpu', allow_download=False) # device='amd', device='intel'

mydir = Path.home() / 'OneDrive' / 'Documents' / 'throawaylien' / 'test' / 'throawaylien.txt'
# C:\Users\joshs\OneDrive\Documents\throawaylien
print("loading directory", mydir)
print(mydir)
loader = TextLoader(mydir.absolute(), autodetect_encoding=True)
print("instantiated loader")
data = loader.load()

# print("Data was: ", data)
print("splitting text and embedding using gpt4all embeddings")
data[0].metadata = {'keywords': 'some random metadata'}
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())
print("finished the vectorestore")
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = client


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("about to invoke the rag_chain")

question = input("Enter your prompt: ")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("just finished invoking the rag_chain")
# cleanup
vectorstore.delete_collection()

