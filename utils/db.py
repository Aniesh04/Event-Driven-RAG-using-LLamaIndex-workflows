from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
# from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings,StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.core import ChatPromptTemplate
from llama_index.core.query_pipeline import QueryPipeline

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import os

from scrapegraphai.graphs import SmartScraperGraph, SearchGraph


os.environ["GOOGLE_API_KEY"] = "AIzaSyBaCPoDRB6V5ixnVCY3QUL0OhOSZwOa1mo"
# os.environ["OPENAI_API_KEY"]= "org-ZBGQ192B8JPnbSGH0mwZZkwS"

prompt = "List me top latest article"


# document = SimpleDirectoryReader(r"C:\Users\CH V N S JAHNAVI\Desktop\vector_database\data", recursive=True).load_data()

documents = SimpleDirectoryReader(r"C:\Users\Anidhinesh\Desktop\Nextgen\knowledge-base", recursive=True).load_data()
# documents = Document(text = "\n\n".join([doc.text for doc in documents]))


# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# embed_model =  LangchainEmbedding(model)
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

nodes = text_splitter.get_nodes_from_documents(
    [Document(text="long text")], show_progress=False
)


chroma_client = chromadb.EphemeralClient()

# chroma_client.delete_collection("quickstart")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection =  db.get_or_create_collection("test")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm = Gemini(model="models/gemini-1.5-flash", temperature=1.0)
# llm = Ollama(model="gemma2:2b")
Settings.llm = llm


index = VectorStoreIndex.from_documents(
    documents,
    # transformations=text_splitter,
    embed_model=embed_model,
    node_parser=nodes,
    storage_context=storage_context,
    llm = llm,
    
)

index.storage_context.persist()


retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=6,
)

template = [
    (
        "system",
        """
        {Prompt}
        * If no prompt is given, repond by saying "Please provide the prompt" *

        Imagine yourself as QA Assistant, tasked with answering questions *only* using the information provided in the given documents or database. You should:
        *Only use* the provided context for your answers.
        - If the answer is *not available* in the context, respond with "web search".
        - If you are given a greeting or casual remark, respond appropriately.
        - *Do not assume* any information that is not explicitly stated in the context. Avoid any interpretation or guessing.
        - Your response should be strictly factual based on the context.

        """,
    ),
    ("user", prompt),
]
prompt_template = ChatPromptTemplate.from_messages(template)


# query_engine = index.as_query_engine(prompt_template=prompt_template, streaming=True).query(prompt)
# print(query_engine)
    


chain = QueryPipeline(chain=[retriever,prompt_template,llm], verbose=False)


if prompt and prompt.strip():
    response = chain.run(prompt)
    result = str(response)[11:]
    # result = "".join([chunk.delta for chunk in response])
    print(result)
else:
    err = """Please provide the prompt"""
    print(err)
 

# def web_response(prompt):
#     graph_config = {
#         "llm": {
#             "model": "ollama/gemma2:2b",
#             # "model": "gemini-1.5-pro",
#             # "api_key": "AIzaSyBaCPoDRB6V5ixnVCY3QUL0OhOSZwOa1mo",
#             "temperature": 0,
#             "format": "json",  # Ollama needs the format to be specified explicitly
#             "base_url": "http://localhost:11434",  # set Ollama URL
#         },
#         "embeddings": {
#             "model": "ollama/nomic-embed-text",
#             # "model": "BAAI/bge-small-en-v1.5"
#             "base_url": "http://localhost:11434",  # set Ollama URL
#         }
#     }
    
#     smart_scraper_graph = SmartScraperGraph(
#     prompt=prompt,
#     # also accepts a string with the already downloaded HTML code
#     source="https://www.espncricinfo.com/",
#     config=graph_config
#     )
    
#     import nest_asyncio
#     nest_asyncio.apply()

#     result = smart_scraper_graph.run()
#     print(result)

# if "web search" in str(response):
#     web_response(prompt)

    
      
