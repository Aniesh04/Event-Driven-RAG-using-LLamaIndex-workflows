from llama_index.core import VectorStoreIndex,SimpleDirectoryReader

# from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings,StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.core import ChatPromptTemplate
from llama_index.core.query_pipeline import QueryPipeline
import qdrant_client
from llama_index.core import Settings

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

from llama_index.core.retrievers import VectorIndexRetriever

import os

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.core import ChatPromptTemplate

from llama_index.core.node_parser import NodeParser
import asyncio

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from llama_index.core import VectorStoreIndex
from typing import Union

from llama_index.utils.workflow import draw_all_possible_flows

from web_scrape import WebSearch

os.environ["GOOGLE_API_KEY"] = "AIzaSyBaCPoDRB6V5ixnVCY3QUL0OhOSZwOa1mo"

class IndexSaveEvent(Event):
  """ Result of Document Loading """
  docs: list
  category: str | None

class ResponseEvent(Event):
  """ Result of Document Loading """
  prompt: str
  index: VectorStoreIndex

class WebSearchEvent(Event):
  category: str

# class StartEvent(Event):
#    category:str

class RAGflow(Workflow):  
  """
    This class is for Local Document Document Retrieval
  """
  category = None
  @step()
  async def Local_Ingest(self, ctx: Context, ev: StartEvent) -> IndexSaveEvent:
      """Load the document"""
      
      dir_path = ev.get("dir_path")
      if not dir_path:
        return "directory path not provided"
      ctx.data["dir_path"] = dir_path

      documents = SimpleDirectoryReader(input_dir=dir_path, recursive=True).load_data()
      ctx.data["docs"] = documents

      prompt = ev.get("prompt")
      ctx.data["prompt"] = prompt

      return IndexSaveEvent(docs=documents, category=None)

  @step(pass_context = True)
  async def Embed_Load(self, ctx: Context, ev: IndexSaveEvent) -> ResponseEvent:
      """Load the document"""
      # documents = ev.get("docs")
      prompt = ctx.data.get("prompt")
      documents = ctx.data.get("docs")
      if not documents:
        return None
      
      documents = [Document(text=doc) if isinstance(doc, str) else doc for doc in documents]

      embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

      text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

      nodes = text_splitter.get_nodes_from_documents(
          [Document(text="long text")], show_progress=False
      )


      chroma_client = chromadb.EphemeralClient()

    #   chroma_client.delete_collection("quickstart")
      db = chromadb.PersistentClient(path="./chroma_db")

      if ev.get("category") is None:
        chroma_collection =  db.get_or_create_collection("Local-Collection")
      else:
        chroma_collection =  db.get_or_create_collection("Web-Collection")

      vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


      storage_context = StorageContext.from_defaults(vector_store=vector_store)


      llm = Gemini(model="models/gemini-1.5-flash", temperature=0.7)
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

      ctx.data["index"] = index
      ctx.data["llm"] = llm
      # ctx.data["prompt"] = prompt

      return ResponseEvent(index=index, prompt= prompt)


  @step(pass_context = True)
  async def Retrieve(self, ctx: Context, ev: ResponseEvent) -> Union[WebSearchEvent,StopEvent]:
      """Retrieve the document"""

      # prompt = ctx.data.get("prompt")
      prompt = ctx.data.get("prompt")
      if not prompt:
        return "Query Not Provided"
      # index = ev.get("index")

      index = ctx.data.get("index")
      if index is None:
            print("Index is empty, load some documents before querying!")
            return None

      llm = ctx.data.get("llm")
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

      chain = QueryPipeline(chain=[retriever,prompt_template,llm], verbose=False)

      response = chain.run(prompt)
      response = str(response)[11:]
      # result = "".join([chunk.delta for chunk in response])
      # print(result)
    # else:
    #     err = """Please provide the prompt"""
    #     print(err)

      if "web search" in response:
        ctx.data["category"] = "web search"
        return WebSearchEvent(category="web search")
      else:
        return StopEvent(result=response)


  @step(pass_context = True)
  async def WebCrawl(self, ctx: Context, ev: WebSearchEvent) -> IndexSaveEvent:

        prompt = ctx.data.get("prompt")
        wb = WebSearch()
        web_docs = await wb.run_search(prompt)
        ctx.data["docs"] = web_docs
        # ctx.data["category"] = "web search"
        return IndexSaveEvent(docs=web_docs, category="web search")
  
  

async def main():
    w = RAGflow(timeout=120, verbose=False)
    result = await w.run(dir_path="./knowledge-base", prompt="Who is Prabhas?")

    print(f"""
    ########################################################################################################################################

    {result}

    ########################################################################################################################################
    """)

if __name__ == "__main__":
    # asyncio.run(main())
    # loop = asyncio.ProactorEventLoop()
    # asyncio.set_event_loop(loop)
    # loop.run_until_complete(main())
    

    draw_all_possible_flows(RAGflow, filename="basic_workflow.html")
    
