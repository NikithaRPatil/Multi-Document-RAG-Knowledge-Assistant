import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def load_documents(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            docs.extend(loader.load())
    return docs

docs = load_documents()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=db.as_retriever()
)

while True:
    query = input("\nAsk: ")
    if query.lower() == "exit":
        break
    print(qa.run(query))
