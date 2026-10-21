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
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, MarkdownNodeParser
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
from llama_index.postprocessor.cohere_rerank import CohereRerank


# LLM Model and embedding for indexing:
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding

embed_model = TogetherEmbedding(model_name="sentence-transformers/msmarco-bert-base-dot-v5")
#Settings.embeddings = embed_model
llm = TogetherLLM(model="mistralai/Mixtral-8x7B-v0.1")
Settings.llm = llm
#Settings.chunk_size = 512
#Settings.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=5) #SentenceSplitter(chunk_size=1024, chunk_overlap=10)


# LLM Model and embedding for evaluation:
from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings

critic_embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
critic_llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1",temperature=0.0,max_tokens=4000,top_k=4)


# Multimodel LLM and embedding functions:
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
#multimodel_embed_model =
multimodel_llm = AnthropicMultiModal(max_tokens=300)


from llama_index.core.indices import MultiModalVectorStoreIndex
from chromadb.utils.data_loaders import ImageLoader
from uuid import uuid4

image_loader = ImageLoader()

PERSIST_DIR = "./RecursiveReteriver"
if not os.path.exists(PERSIST_DIR):
   # load documents
   #parser = LlamaParse(result_type="markdown",verbose=True)
   #documents = parser.load_data("./APP_Docs/ETSI.pdf")
   documents = SimpleDirectoryReader("./APP_Docs").load_data(num_workers=1)
   # Parsing
   transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=5)]#[MarkdownNodeParser()] #[MarkdownElementNodeParser(llm = llm, num_workers=8)]#[TokenTextSplitter(chunk_size=512, chunk_overlap=5)] #,TitleExtractor(nodes=20),SummaryExtractor(summaries=["prev", "self"]),KeywordExtractor(keywords=10)] #SentenceSplitter(chunk_size=1024, chunk_overlap=10)
   pipeline = IngestionPipeline(transformations=transformations)
   base_nodes = pipeline.run(documents=documents)
   # set node ids to be a constant
   for idx, node in enumerate(base_nodes):
       node.id_ = f"node-{idx}"

   sub_chunk_sizes = [128]
   sub_node_parsers = [SentenceSplitter(chunk_size=c, chunk_overlap=2) for c in sub_chunk_sizes]
   all_nodes = []
   for base_node in base_nodes:
       for n in sub_node_parsers:
           sub_nodes = n.get_nodes_from_documents([base_node])
           sub_inodes = [IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes]
           all_nodes.extend(sub_inodes)
       # also add original node to node
       original_node = IndexNode.from_text_node(base_node, base_node.node_id)
       all_nodes.append(original_node)

   all_nodes_dict = {n.node_id: n for n in all_nodes}
   # save to disk
   chroma_client = chromadb.PersistentClient(path="./RecursiveReteriver")
   chroma_collection = chroma_client.get_or_create_collection("rag_RecursiveReteriver")
   vector_store = ChromaVectorStore(chroma_collection=chroma_collection,data_loader=image_loader)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   #index = MultiModalVectorStoreIndex(nodes=all_nodes,storage_context=storage_context,embed_model=embed_model)
   index = VectorStoreIndex(nodes=all_nodes,storage_context=storage_context,embed_model=embed_model) 
else:
   # load from disk
   chroma_client2 = chromadb.PersistentClient(path="./RecursiveReteriver")
   chroma_collection = chroma_client2.get_or_create_collection("rag_RecursiveReteriver")
   vector_store = ChromaVectorStore(chroma_collection=chroma_collection,data_loader=image_loader)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   #index = MultiModalVectorStoreIndex.from_vector_store(vector_store,storage_context,embed_model=embed_model)
   index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model) 
   
# RAG without Parser Output
from llama_index.legacy.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.query_pipeline.query import InputKeys,OutputKeys
from typing import Dict, Any
from pydantic import Field

cohere_rerank = CohereRerank(api_key=os.environ["COHERE_API_KEY"], top_n=2)

retriever = index.as_retriever(similarity_top_k=4, image_similarity_top_k=1, node_postprocessors=[cohere_rerank])
recursive_retriever = RecursiveRetriever("vector",retriever_dict={"vector": retriever},node_dict=all_nodes_dict,verbose=True) 

# Setup LLMLingua
#from llama_index.legacy.postprocessor import LongLLMLinguaPostprocessor

#node_postprocessor = LongLLMLinguaPostprocessor(device_map='auto',instruction_str="Given the context, please answer the final question",target_token=100,rank_method="longllmlingua",
#                                                additional_compress_kwargs={"condition_compare": True,"condition_in_question": "after","context_budget": "+100","reorder_context": "sort","dynamic_context_compression_ratio": 0.3})

#new_retrieved_nodes = node_postprocessor.postprocess_nodes(retrieved_nodes)

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
        nodes = recursive_retriever.retrieve(kwargs["query_str"])
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

p_adv = QueryPipeline(verbose=True)
p_adv.add_modules({"input": InputComponent(),"retriever": retriever_component,"prompt":prompt_tmpl,"llm":llm})

p_adv.add_link("input", "retriever", dest_key="query_str")
p_adv.add_link("input", "prompt", dest_key="query_str")
p_adv.add_link("retriever","prompt", dest_key="context_str")
p_adv.add_link("prompt","llm", dest_key='messages')   


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
    nodes = recursive_retriever.retrieve(input_prompt)
    #new_retrieved_nodes = node_postprocessor.postprocess_nodes(nodes)
    #context = "\n\n".join([n.get_content() for n in new_retrieved_nodes])
    context = "\n\n".join([n.get_content() for n in nodes])
    contexts.append([context])
    output = p_adv.run(input=input_prompt)
    answer.append(output.message.content) #output.response

print("answer",answer)


################################################
from llama_index.evaluation.tonic_validate import TonicValidateEvaluator
questions = question
llm_answers = answer
retrieved_context_lists = contexts
reference_answers = ground_truth

tonic_validate_evaluator = TonicValidateEvaluator(model_evaluator="mistralai/Mixtral-8x7B-v0.1")
scores = tonic_validate_evaluator.evaluate_run([questions], [llm_answers], [retrieved_context_lists], [reference_answers])

print(scores.run_data[0].scores)
print(scores.overall_scores)

validate_api = ValidateApi()
project_id = "API"
validate_api.upload_run(project_id, scores)
################################################

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

