
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
import streamlit as st
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor


# Set up the LLM and Embedding model
def Initialize_LLM_embeddingsModels(llm, embed_model):
    Settings.llm = Ollama(model=llm, request_timeout=60.0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name= embed_model
    )

# Load data
def LoadDocument(doc_path):
    documents = SimpleDirectoryReader(
        input_files=[doc_path]
    ).load_data()

    return documents


def RagSetup(documents, useNodes):
    # Connect to your qdrant instance
    client = qdrant_client.QdrantClient(
        # you can use :memory: mode for fast and light-weight experiments,
        # it does not require to have Qdrant deployed anywhere
        # but requires qdrant-client >= 1.1.1
        location=":memory:"
    )

    index_name = "MyExternalContext"

    # Construct vector store
    vector_store = QdrantVectorStore(client=client, collection_name=index_name, enable_hybrid=True)

    node_parser = SentenceSplitter(chunk_size=256, paragraph_separator="\n\n", )

    nodes = node_parser.get_nodes_from_documents(documents)

    title_extractor = TitleExtractor(nodes=5)


    # Set up the storage for the embeddings
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup the index
    # build VectorStoreIndex that takes care of chunking documents
    # and encoding chunks to embeddings for future retrieval

    if useNodes:
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, transformations=[title_extractor])

    else:
        index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)
    
    # The target key defaults to `window` to match the node_parser's default
    postproc = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    # BAAI/bge-reranker-base
    # link: https://huggingface.co/BAAI/bge-reranker-base
    rerank = SentenceTransformerRerank(
        top_n = 2, 
        model = "BAAI/bge-reranker-base"
    )


    query_engine = index.as_query_engine(
        similarity_top_k = 2, 
        vector_store_query_mode="hybrid", 
        alpha=0.5,
        node_postprocessors = [postproc, rerank]
        # streaming=True
    )

    qa_prompt_tmpl_str = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "---------------------\n"
        "Question: {query_str}"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Answer: "
    )


    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )

    return query_engine


def Main():

    Initialize_LLM_embeddingsModels("mistral:7b-instruct-q4_0", "BAAI/bge-small-en-v1.5")
    documents = LoadDocument("./data/RAG Input Doc.pdf")

    usenodes = False

    query_engine = RagSetup(documents,usenodes)
 
    print('Questions and their Responses for Rag Input Doc: \n\n')
    print('Q1: Which paper received the highest number of stars per hour?'+'\n')
    streaming_response  = query_engine.query(
        "Which paper received the highest number of stars per hour?"
    )
    print("Response: "+streaming_response.response)
    print()

    print("Q2: What is the focus of the 'MeshAnything' project?"+'\n')
    streaming_response  = query_engine.query(
        "What is the focus of the 'MeshAnything' project?"
    )
    print("Response: "+streaming_response.response)
    print()

    print("Q3: Which paper discusses the integration of Large Language Models with Monte Carlo Tree Search?"+'\n')
    streaming_response  = query_engine.query(
        "Which paper discusses the integration of Large Language Models with Monte Carlo Tree Search?"
    )
    print("Response: "+streaming_response.response)
    print()

    print("Q4: What advancements does the 'VideoLLaMA 2' paper propose?"+'\n')
    streaming_response  = query_engine.query(
        "What advancements does the 'VideoLLaMA 2' paper propose?"
    )
    print("Response: "+streaming_response.response)
    print()

    print("Q5: Which paper was published most recently?"+'\n')
    streaming_response  = query_engine.query(
        "Which paper was published most recently?"
    )
    print("Response: "+streaming_response.response)
    print()

    print("Q6: Identify a paper that deals with language modeling and its scalability."+'\n')
    streaming_response  = query_engine.query(
        "Identify a paper that deals with language modeling and its scalability."
    )
    print("Response: "+streaming_response.response)
    print()

    print("Q7: Which paper aims at improving accuracy in Google-Proof Question Answering?"+'\n')
    streaming_response  = query_engine.query(
        "Which paper aims at improving accuracy in Google-Proof Question Answering?"
    )
    print("Response: "+streaming_response.response)
    print()

    print("""Q8: List the categories covered by the paper titled 'TextGrad: Automatic "Differentiation" via Text'."""+'\n')
    streaming_response  = query_engine.query(
        """List the categories covered by the paper titled 'TextGrad: Automatic "Differentiation" via Text'."""
    )
    print("Response: "+streaming_response.response)
    print()

########### Extra Credit 1###########
    print("""Q9 Can you summarize the 'TextGrad: Automatic "Differentiation" via Text' paper?""" +'\n')
    streaming_response  = query_engine.query(
        """Can you summarize the 'TextGrad: Automatic "Differentiation" via Text' paper?"""
    )
    print("Response: "+streaming_response.response)
    print()  

    print("""Q10  Can you give me details of the paper authored by trotsky1997/mathblackbox?""" +'\n')
    streaming_response  = query_engine.query(
        """Can you give me details of the paper authored by trotsky1997/mathblackbox?"""
    )
    print("Response: "+streaming_response.response)
    print() 

########### Extra Credit 2###########
    documents = LoadDocument("./data/paul_graham_essay.txt")

    usenodes = True

    query_engine = RagSetup(documents,usenodes)

    print('Questions and their Responses for paul_graham_essay: \n\n')
    print("What was Paul Graham's initial career plan while he was in a PhD program?"+'\n')
    streaming_response  = query_engine.query(
        "What was Paul Graham's initial career plan while he was in a PhD program?"
    )
    print("Response: "+streaming_response.response)
    print()
    print("""Correct Response: Paul Graham was in a PhD program in computer science but planning to be an artist. He was also working on Lisp hacking and his project "On Lisp"""+'\n')

    print("Q2: Where did Paul Graham go after being accepted to art school?"+'\n')
    streaming_response  = query_engine.query(
        "Where did Paul Graham go after being accepted to art school?"
    )
    print("Response: "+streaming_response.response)
    print()
    print("""Correct Response: Paul Graham went to the Rhode Island School of Design (RISD) after being accepted into their BFA program"""+'\n')

    print("Q3: What was unique about the entrance exam at the Accademia di Belli Arti in Florence?"+'\n')
    streaming_response  = query_engine.query(
        "What was unique about the entrance exam at the Accademia di Belli Arti in Florence?"
    )
    print("Response: "+streaming_response.response)
    print()
    print("""Correct Response: Only foreign students had to take the entrance exam, which might have been a way to limit the number of foreign students because many were attracted to study art in Florence"""+'\n')

if __name__ == "__main__":
    Main()


