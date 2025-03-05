import time
import streamlit as st
import os
import fitz
from langchain.tools import Tool
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from PIL import ImageDraw
from pdf2image import convert_from_path

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
                    "page": page_num,
                    "bounding_box": bbox
                })
    doc.close()
    return chunks, page_datas, "\n".join(full_text)

st.session_state.gemini_key = st.secrets.gemini_api_key

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""

def setKey():
    st.session_state.gemini_key = st.session_state.gemini_key_field

st.set_page_config(layout="wide")

if st.session_state.gemini_key == "":
    st.text_input("Gemini API Key", type="default", key="gemini_key_field", on_change=setKey)
elif not st.session_state.loaded:
    st.header("Talk with my PDF")
    uploaded_file = st.file_uploader("Please upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        tmp_location = 'tmp_' + str(int(time.time()))
        with open(tmp_location, "wb") as f:
            f.write(uploaded_file.read())

        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_key

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        vector_store = InMemoryVectorStore(embedding_model)

        graph_builder = StateGraph(MessagesState)

        text_data, page_datas, full_text = extract_text_with_metadata(tmp_location)

        def retrieve(query: str):
            print('vector search tool access: ' + query)
            
            search_results = vector_store.similarity_search_with_score(query, k=10)
            retrieved_docs_filtered = []
            for search_result in search_results:
                if (search_result[1] > 0.5):
                    retrieved_docs_filtered.append(search_result[0])
            
            serialized = "\n\n".join(
                (f"Id: {doc.id} Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs_filtered
            )
            return serialized, retrieved_docs_filtered

        def full_document_tool(query: str):
            print('full doc tool access')
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

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""

            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Your context is a document "
                "Use your tools as context to answer, prefer your Vector Search Tool when "
                "you are asked for specific information in the document and your Full Document Tool when you will need more context"
                "as when you have to summarize the document. "
            )

            llm_with_tools = llm.bind_tools([vector_search, full_doc])
            prompt = [SystemMessage(system_message_content)]
            response = llm_with_tools.invoke(prompt + state["messages"])
            return {"messages": [response]}

        tools = ToolNode([vector_search, full_doc])

        def generate(state: MessagesState):
            """Generate answer."""
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]

            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. You are a multilingual assistant. Always respond in the same language as the user\'s request."
                "If you found the required information in the provided sources, always tell the user on which page numbers they are(incremented by 1 as they are zero-based) "
                "\n\n"
                f"{docs_content}"
            )
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages

            response = llm.invoke(prompt)
            return {"messages": [response]}

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

        memory = MemorySaver()

        graph = graph_builder.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": "abc123"}}

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
        print("stored in vector store: " + str(len(document_ids)))
        st.session_state.pdfimages = convert_from_path(tmp_location, size=(700, None))
        st.session_state.loaded = True
        st.session_state.graph = graph
        st.session_state.cff = config
        st.session_state.page_datas = page_datas
        st.session_state.artifacts = []
        st.session_state.messages = []
        os.remove(tmp_location)
        st.rerun()
else:
    def reset():
        st.session_state.loaded = False

    spaceStart, col1, col2, spaceEnd = st.columns([1, 3, 5, 1])
    with col1:
        st.subheader('Chat')
        container1 = st.container()
        with container1:
            containerInnerUp = st.container(height=400)
            with containerInnerUp:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                    # Create a chat input field to allow the user to enter a message. This will display
                    # automatically at the bottom of the page.
            containerInnerDown = st.container(height=73)
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
            st.button("Start New Chat", on_click=reset)
    with col2:     
        st.subheader('Document')
        pdfContainer = st.container(height=600)
        with pdfContainer:
            for i, image in enumerate(st.session_state.pdfimages):
                imgCopy = image.copy()
                draw = ImageDraw.Draw(imgCopy)
                mediaBox = st.session_state.page_datas[i]
                pageWidth = mediaBox[2] - mediaBox[0]
                pageHeight = mediaBox[3] - mediaBox[1]
                scale = image.width / pageWidth
                for artifact in st.session_state.artifacts:
                    if (artifact.metadata["page"] == i):
                        origBB = artifact.metadata["bounding_box"]
                        scaledBB = (origBB[0] * scale, origBB[1] * scale, origBB[2] * scale, origBB[3] * scale)
                        draw.rectangle(scaledBB, outline="red", width=3)
                st.image(imgCopy)
               



