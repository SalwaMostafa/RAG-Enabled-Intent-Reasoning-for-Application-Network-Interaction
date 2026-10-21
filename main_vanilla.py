__import__('pysqlite3')
import sys
import time
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import nest_asyncio
nest_asyncio.apply()
import os
import chromadb
import copy
import json
import pandas as pd
from typing import List, Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llama_dataset import LabelledRagDataset
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from IPython.display import Markdown, display
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from llama_index.core import Settings
from llama_index.llms.together import TogetherLLM
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.extractors import (SummaryExtractor,QuestionsAnsweredExtractor)
from llama_index.core.evaluation import (generate_question_context_pairs,EmbeddingQAFinetuneDataset)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (SummaryExtractor,QuestionsAnsweredExtractor,TitleExtractor,KeywordExtractor,BaseExtractor)
from llama_index.core.schema import MetadataMode
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_parse import LlamaParse



# LLM Model and embedding for indexing:
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding

from llama_index.legacy.llms.llama_cpp import LlamaCPP

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
#    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.0,
    max_new_tokens=100,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=1000,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    #callback_manager=callback_manager,#CallbackManager,
    verbose=True,
)

embed_model = TogetherEmbedding(model_name="sentence-transformers/msmarco-bert-base-dot-v5")
Settings.embeddings = embed_model
#llm = TogetherLLM(model="meta-llama/Llama-2-7b-hf") #mistralai/Mixtral-8x7B-v0.1
Settings.llm = llm
Settings.chunk_size = 128
Settings.text_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=10)

#from llama_index.embeddings.mistralai import MistralAIEmbedding
#model_name = "mistral-embed"
#embed_model = MistralAIEmbedding(model_name=model_name)

# LLM Model and embedding for evaluation:
from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings

critic_embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
critic_llm = Together(model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",temperature=0.0,max_tokens=4000,top_k=2)


# Multimodel LLM and embedding functions:
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
#multimodel_embed_model =
multimodel_llm = AnthropicMultiModal(max_tokens=300)


# Vanilla RAG

PERSIST_DIR = "./vanilla_chroma_db"
if not os.path.exists(PERSIST_DIR):
   # load documents
   #parser = LlamaParse(result_type="markdown",verbose=True)
   #documents = parser.load_data("./APP_Docs")
   documents = SimpleDirectoryReader("./APP_Docs").load_data(num_workers=1)
   # Parsing
   transformations = [SentenceSplitter(chunk_size=128, chunk_overlap=5)] #[MarkdownElementNodeParser(llm = llm, num_workers=8),TitleExtractor(nodes=20),SummaryExtractor(summaries=["prev", "self"]),KeywordExtractor(keywords=10)] # SentenceSplitter(chunk_size=1024, chunk_overlap=10) TokenTextSplitter(chunk_size=512, chunk_overlap=10) 
   pipeline = IngestionPipeline(transformations=transformations)
   base_nodes = pipeline.run(documents=documents) #node_parser.get_nodes_from_documents(documents)
   # set node ids to be a constant
   for idx, node in enumerate(base_nodes):
       node.id_ = f"node-{idx}"
   # save to disk
   chroma_client = chromadb.PersistentClient(path="./vanilla_chroma_db")
   chroma_collection = chroma_client.get_or_create_collection("rag_llamaindex")
   vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   index = VectorStoreIndex(base_nodes, storage_context=storage_context,embed_model=embed_model)
else:
   # load from disk
   chroma_client2 = chromadb.PersistentClient(path="./vanilla_chroma_db")
   chroma_collection = chroma_client2.get_or_create_collection("rag_llamaindex")
   vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
   #storage_context = StorageContext.from_defaults(vector_store=vector_store) ,storage_context
   index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)
   
   
# RAG without Parser Output
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.query_pipeline.query import InputKeys,OutputKeys
from typing import Dict, Any
from pydantic import Field

retriever = index.as_retriever(similarity_top_k=2)

class PostprocessingComponent(CustomQueryComponent):
    """Retrieved nodes are postprocessing to get the context."""

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"query_str"}

    @property
    def _output_keys(self) -> set:
        return {"output"}

    def _run_component(self, **kwargs):
        nodes = retriever.retrieve(kwargs["query_str"])
        context_str = "\n\n".join([n.get_content() for n in nodes])
        return {"output": context_str}

retriever_component = PostprocessingComponent()


prompt_str  = ("We have provided context information below. \n"
               "---------------------\n"
               "{context_str}"
              "\n---------------------\n"
              "Given this information, Please provide the performance recommendations metrics and their values to the scenario {query_str} in JSON formate.\n")

prompt_tmpl = PromptTemplate(prompt_str)


class KPI(BaseModel):
    """Object representing different performance metrics."""
    metric : str = Field(..., description="specific performance metric.")
    value : int or float or str = Field(..., description="Value of the specific performance metric along with the unit.")

class Scenario(BaseModel):
    """Object representing a list of performance recommendations to different traffic scenarios."""
    Performance_recommendations: List[KPI] = Field(..., description="List of performance recommendations metrics.")

out_prompt_str = "Please provided the given information {input} rewrite the list of performance recommendations in JSON formate and if some values are not available write None."

output_parser = PydanticOutputParser(Scenario)
json_prompt_str = output_parser.format(out_prompt_str)
json_prompt_tmpl = PromptTemplate(json_prompt_str)

p = QueryPipeline(verbose=True)
p.add_modules({"input": InputComponent(),"retriever": retriever_component,"prompt":prompt_tmpl,"llm":llm})

p.add_link("input", "retriever", dest_key="query_str")
p.add_link("input", "prompt", dest_key="query_str")
p.add_link("retriever","prompt", dest_key="context_str")
p.add_link("prompt","llm", dest_key='messages')   

# Dataset for evaluation using RAGAS   
   
query_engine = index.as_query_engine()
response = query_engine.query("List all the applications")
print("response",response)

question = ["4K On Demand Video",
            "4K Live Video",
            "4K 360-degree VR panoramic video (VR IMAX and VR concert) All in one VR machine" ,
            "3K Cloud VR (Game)"]

ground_truth = ["Data Rate/Throughput (downlink):30 Mbps , Delay:RTT < 100 ms, Packet Loss Rate:10-3 , Resolution: 4K, Coverage Level CSI RSRP:-113 dBm , Coverage Quality CSI SINR:-2 dB",
                "Data Rate/Throughput (downlink): 40 Mbps, Delay:RTT < 100 ms, Packet Loss Rate:10-3 , Resolution: 4K, Coverage Level CSI RSRP:-110 dBm , Coverage Quality CSI SINR:-1 dB",
                "Data Rate/Throughput (downlink):80 Mbps , Delay:RTT < 100 ms, Packet Loss Rate:10-3 , Resolution: 4K, Coverage Level CSI RSRP:-108 dBm , Coverage Quality CSI SINR:1 dB",
                "Data Rate/Throughput (downlink): 100 Mbps, Delay:RTT < 25 ms, Packet Loss Rate:10-3 (TCP) 10-2 (UDP), Resolution: 3K, Coverage Level CSI RSRP:-107 dBm , Coverage Quality CSI SINR:2 dB"]

answer = []
contexts = []

for input_prompt in question:
    context = []
    nodes = retriever.retrieve(input_prompt)
    context = "\n\n".join([n.get_content() for n in nodes])
    contexts.append([context])
    output = p.run(input=input_prompt)
    answer.append(output.message.content) #output.response


from ragas.metrics import (answer_relevancy,context_precision,context_recall,faithfulness,answer_correctness)
from datasets import Dataset
# To dict
data = {"question": question,"answer": answer,"contexts": contexts,"ground_truth":ground_truth}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

from llama_index.llms.huggingface import (HuggingFaceInferenceAPI,HuggingFaceLLM)
from ragas import evaluate


result_context_precision = evaluate(dataset, metrics=[context_precision], llm=critic_llm, embeddings=critic_embeddings)
result_context_precision.to_pandas()

print(result_context_precision)


result_context_recall = evaluate(dataset, metrics=[context_recall], llm=critic_llm, embeddings=critic_embeddings) #answer_relevancy,answer_correctness
result_context_recall.to_pandas()

print(result_context_recall)

result_answer_correctness = evaluate(dataset, metrics=[answer_correctness], llm=critic_llm, embeddings=critic_embeddings) #answer_relevancy,answer_correctness
result_answer_correctness.to_pandas()

print(result_answer_correctness)

result_answer_relevancy = evaluate(dataset, metrics=[answer_relevancy], llm=critic_llm, embeddings=critic_embeddings) #answer_relevancy,answer_correctness
result_answer_relevancy.to_pandas()

print(result_answer_relevancy)


result_faithfulness = evaluate(dataset, metrics=[faithfulness], llm=critic_llm, embeddings=critic_embeddings) #answer_relevancy,answer_correctness
result_faithfulness.to_pandas()

print(result_faithfulness)