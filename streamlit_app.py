import time
from PIL import Image, ImageDraw
import streamlit as st
from langchain.tools import Tool
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
from langchain.schema import Document
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdfplumber
import streamlit as st
import os
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace, NumericNamespace
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_vertexai import VectorSearchVectorStore
# import redis
from langchain_redis import RedisConfig, RedisVectorStore
from redisvl.query.filter import Tag, Num, Text
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
import glob

# Step 1: Extract text and metadata from PDF
def extract_text_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    page_datas = []
    full_text = []
    for page_num, page in enumerate(doc):
        page_datas.append(page.rect)
        full_text.append(page.get_text())
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                # Combine all lines in the block into one text chunk for better context
                block_lines = []
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    block_lines.append(line_text)
                block_text = " ".join(block_lines)
                bbox = block.get("bbox", None)
                chunks.append({
                    "text": block_text,
                    "page": page_num + 1,
                    "bounding_box": bbox
                })
    doc.close()
    return chunks, page_datas, "\n".join(full_text)


def delete_files_with_prefix(prefix, directory="."):
    """Deletes all files starting with a given prefix in a directory."""
    file_pattern = os.path.join(directory, f"{prefix}*")  # Match prefix
    files_to_delete = glob.glob(file_pattern)  # Find matching files

    for file_path in files_to_delete:
        try:
            os.remove(file_path)  # Delete file
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Example: Delete all images with prefix "temp_page_"


# from redissaver import RedisSaver

# redisCli = redis.StrictRedis(host='localhost', port=6379, password='krasi')

# CREDENTIALS

st.set_page_config(layout="wide")
# Show title and description.

st.header("💬 Krasi Test")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )

def setKey():
    st.session_state.gemini_key = st.session_state.gkey

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""

if "artifacts" not in st.session_state:
    st.session_state.artifacts = []

print('reruns')
if st.session_state.gemini_key == "":
    st.text_input("API Key", type="default", key="gkey", on_change=setKey)
elif not st.session_state.loaded:
    print(st.session_state.gemini_key)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file is not None:
        delete_files_with_prefix("temp_page_")
        tmp_location = 'tempKrasiPdf' + str(int(time.time()))
        with open(tmp_location, "wb") as f:
            f.write(uploaded_file.read())

        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_key

        # LLVM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # EMBEDDINGS
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004")

        # VECTOR STORE
        # config = RedisConfig(
        #     index_name="indexKrasi_",
        #     redis_client=redisCli,
        #     metadata_schema=[
        #         {"name": "color", "type": "text"},
        #         {"name": "price", "type": "numeric"},
        #         {"name": "season", "type": "tag"}
        #     ]
        # )

        # vector_store = RedisVectorStore(embeddings=embedding_model,config=config)

        vector_store = InMemoryVectorStore(embedding_model)

        filters = [Namespace(name="season", allow_tokens=["winter"])]

        # redis filters
        season = Tag("season") == "spring"
        price = Num("price") < 200
        color = Text("color") == "blue"
        redisFilter = color

        # filters = []
        # CHAT
        graph_builder = StateGraph(MessagesState)

        text_data, page_datas, full_text = extract_text_with_metadata(tmp_location)

        # @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            print('vector search tool access')
            print("k1")
            print(query)
            print("k2")
            
            retrieved_docs_1 = vector_store.similarity_search_with_score(query, k=10)
            # print (f"krkrkr: ", len(retrieved_docs_first))
            # retrieved_docs= []
            # for doc in retrieved_docs_first:
            #     print("adkakdkas")
            #     print(doc[0])
            #     retrieved_docs.append(doc[0])
            retrieved_docs = []
            for doc in retrieved_docs_1:
                if (doc[1] > 0.5):
                    retrieved_docs.append(doc[0])
            
            serialized = "\n\n".join(
                (f"Id: {doc.id} Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        def full_document_tool(query: str):
            print('full doc tool access')
            print(full_text)
            return full_text

        vector_search = Tool(
            name="retrieve",
            func=retrieve,
            description=(
                "Use this tool when the user's question requires retrieving relevant "
                "chunks or specific sections from document. This performs a "
                "similarity search to find the best matches for the query."
            ),
            response_format="content_and_artifact"
        )

        full_doc = Tool(
            name="full_document_tool",
            func=full_document_tool,
            description=(
                "Use this tool if the question explicitly requires the *entire* document, "
                "or if the user asks for a full article/text rather than just a snippet."
            ),
        )

        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""

            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Your context is a document "
                "Use your tools as context to answer, prefer your Vector Search Tool when "
                "you are asked for specific information in the document and your Full Document Tool when you will need more context"
                "as when you have to summarize the document "
                "don't know. If the user asks about his document use your retrieval tools to search."
            )

            llm_with_tools = llm.bind_tools([vector_search, full_doc])
            prompt = [SystemMessage(system_message_content)]
            response = llm_with_tools.invoke(prompt + state["messages"])
            # MessagesState appends messages to state instead of overwriting
            return {"messages": [response]}

        # Step 2: Execute the retrieval.

        # Step 3: Generate a response using the retrieved content.
        def generate(state: MessagesState):
            """Generate answer."""
            # Get generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]

            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise. If the user asks about his document use your retrieval tool to search."
                "\n\n"
                f"{docs_content}"
            )
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)
                      ] + conversation_messages

            # Run
            response = llm.invoke(prompt)
            return {"messages": [response]}

    
        tools = ToolNode([vector_search, full_doc])
        
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        # MEMORY
        # memory = RedisSaver(conn=redisCli)
        memory = MemorySaver()

        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "abc123"}}

        # loader = PyPDFLoader(tmp_location)
        # docs = loader.load()
        # print(f"doc length: {len(docs[0].page_content)}.")
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000, chunk_overlap=200, add_start_index=True)
        # all_splits = text_splitter.split_documents(docs)
        # print(f"split into {len(all_splits)} sub-documents.")
        # print(all_splits[0])
        # # INDEX CHUNKS
        # document_ids = vector_store.add_documents(documents=all_splits)
        # print("stored in vector store.")
        # print(document_ids)

        # text_data = []
        # page_datas = []
        # with pdfplumber.open(tmp_location) as pdf:
        #     for page_num, page in enumerate(pdf.pages):
        #         words = page.extract_words()  # Extract words with bounding boxes
        #         page_datas.append(page.bbox)
        #         paragraphs = []
        #         current_paragraph = []
        #         prev_y = None
        #         y_threshold = 30  # Adjust this based on paragraph spacing

        #         for word in words:
        #             x0, y0, x1, y1, text = word["x0"], word["top"], word["x1"], word["bottom"], word["text"]

        #             # Detect paragraph breaks based on vertical spacing
        #             if prev_y is not None and abs(y0 - prev_y) > y_threshold:
        #                 if current_paragraph:
        #                     paragraphs.append(current_paragraph)
        #                 current_paragraph = []

        #             current_paragraph.append((x0, y0, x1, y1, text))
        #             prev_y = y0

        #         if current_paragraph:  # Append last paragraph
        #             paragraphs.append(current_paragraph)

        #         # Convert paragraphs into structured text + bounding boxes
        #         for paragraph in paragraphs:
        #             paragraph_text = " ".join([w[4] for w in paragraph])  # Join words into paragraph
        #             bbox = (
        #                 min(w[0] for w in paragraph),  # x0
        #                 min(w[1] for w in paragraph),  # y0
        #                 max(w[2] for w in paragraph),  # x1
        #                 max(w[3] for w in paragraph),  # y1
        #             )

        #             text_data.append({
        #                 "text": paragraph_text,
        #                 "page": page_num + 1,
        #                 "bounding_box": bbox
        #             })


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True, separators=["\n\n"])
        documents = []
        for entry in text_data:
            chunks = text_splitter.split_text(entry["text"])
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={
                    "page":entry["page"],
                    "bounding_box":entry["bounding_box"]
                }))
        
        document_ids = vector_store.add_documents(documents=documents)
        print("stored in vector store.")
        print(document_ids)
        st.session_state.pdfimages = convert_from_path(tmp_location, size=(700, None))

        st.session_state.loaded = True
        st.session_state.graph = graph
        st.session_state.cff = config
        st.session_state.page_datas = page_datas
        os.remove(tmp_location)
        st.rerun()

else:
    def reset():
        st.session_state.loaded = False
        st.session_state.artifacts = []
        st.session_state.pdfimages = []
        st.session_state.messages = []

    st.button('New Chat', on_click=reset)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('Chat')
        container1 = st.container(height=600)
        with container1:
            # st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
            if "messages" not in st.session_state:
                st.session_state.messages = []
                # Display the existing chat messages via `st.chat_message`.
            containerInnerUp = st.container(height=400)
            with containerInnerUp:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                    # Create a chat input field to allow the user to enter a message. This will display
                    # automatically at the bottom of the page.
            containerInnerDown = st.container(height=100)
            with containerInnerDown:
                if prompt := st.chat_input("What is up?"):
                    # Store and display the current prompt.
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt})
                    with containerInnerUp:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    for step in st.session_state.graph.stream(
                        {"messages": [{"role": "user", "content": prompt}]},
                        stream_mode="values",
                        config=st.session_state.cff,
                    ):
                        msg = step["messages"][-1]
                        artifacts = []
                        if (len(step["messages"]) > 1):
                            recent_tool_message = step["messages"][-2]
                            if recent_tool_message.type == "tool" and recent_tool_message.artifact:
                                for artifact in recent_tool_message.artifact:
                                    artifacts.append(artifact)
                        st.session_state.artifacts = artifacts

                        if msg.type == 'ai' and not msg.tool_calls:
                            # with st.chat_message("assistant"):
                                # st.write(msg.content)
                            st.session_state.messages.append(
                                    {"role": "assistant", "content": msg.content})
                    st.rerun()
            
    with col2:     
        st.header('Document')
        st.markdown("""
            <style>
                .block-container {
                    padding: 0rem 0rem !important;
                }
            </style>
            """, unsafe_allow_html=True)
        pdfContainer = st.container(height=600)
        static_width = 700
        with pdfContainer:
            for i, image in enumerate(st.session_state.pdfimages):
                imgCopy = image.copy()
                draw = ImageDraw.Draw(imgCopy)
                mediaBox = st.session_state.page_datas[i]
                pageWidth = mediaBox[2] - mediaBox[0]
                pageHeight = mediaBox[3] - mediaBox[1]
                scale = image.width / pageWidth
                for artifact in st.session_state.artifacts:
                    if (artifact.metadata["page"] == i + 1):
                        origBB = artifact.metadata["bounding_box"]
                        scaledBB = (origBB[0] * scale, origBB[1] * scale, origBB[2] * scale, origBB[3] * scale)
                        draw.rectangle(scaledBB, outline="red", width=3)
                        print(artifact)
                st.image(imgCopy, width=static_width)
               



