def find_doc(question):
    """
    A find_doc function to query for specific results at the user's request.

    Args:
        question (str): The query to execute.

    Returns:
        response [str]: The text content of the results of the search.
    """
    from pathlib import Path
    import chromadb
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
    from langchain import hub
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    if not isinstance(question, str):
        return "Error: expected query to be type STR, but got ", type(question), " value: ", question

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    fulldir = Path.home() / 'OneDrive' / 'Documents' / 'throawaylien'
    text_loader_kwargs={'autodetect_encoding': True}
    dirloader = DirectoryLoader(fulldir.absolute(), glob='**/*.txt', loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    dirdata = dirloader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    splits = text_splitter.split_documents(dirdata)

    embeddings = OpenAIEmbeddings(
        base_url="http://localhost:8000/v1",
        api_key="n/a",
        model="nomic-ai/nomic-embed-text-v1.5-GGUF",
        )
    new_client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    template = """Use the provided pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible.

    CONTEXT:

    [[[
    {context}
    ]]]

    QUESTION: {question}

    HELPFUL ANSWER:"""
    custom_rag_prompt = PromptTemplate.from_template(template)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | client
        | StrOutputParser()
    )
    
    # chunk = ''
    # for chunk in rag_chain.stream(question):
    #     chunk += chunk
    # print("Chunk was: ", chunk)

    result = rag_chain.invoke(question)
    return result
