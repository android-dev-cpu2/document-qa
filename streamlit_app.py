from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os
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

# from redissaver import RedisSaver

# redisCli = redis.StrictRedis(host='localhost', port=6379, password='krasi')

# CREDENTIALS


# Show title and description.
st.title("💬 Krasi Test")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )

if "loaded" not in st.session_state:
   st.session_state.loaded = False

print('reruns')
if not st.session_state.loaded:
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file is not None:
       tmp_location = 'tempKrasiPdf'
       with open(tmp_location, "wb") as f:
           f.write(uploaded_file.read())
    
       if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = st.secrets.gemini_api_key

       # LLVM
       llm = ChatGoogleGenerativeAI(
           model="gemini-1.5-pro",
           temperature=0,
           max_tokens=None,
           timeout=None,
           max_retries=2,
       )
    
       # EMBEDDINGS
       embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
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
    
       @tool(response_format="content_and_artifact")
       def retrieve(query: str):
           """Retrieve information related to a query."""
           print("k1")
           print(query)
           print("k2")
           retrieved_docs = vector_store.similarity_search(query, k=4)
           serialized = "\n\n".join(
               (f"Id: {doc.id} Source: {doc.metadata}\n" f"Content: {doc.page_content}")
               for doc in retrieved_docs
           )
           return serialized, retrieved_docs
    
       # Step 1: Generate an AIMessage that may include a tool-call to be sent.
       def query_or_respond(state: MessagesState):
           """Generate tool call for retrieval or respond."""

           system_message_content = (
               "You are an assistant for question-answering tasks. "
               "Use the following pieces of retrieved context to answer "
               "the question. If you don't know the answer, say that you "
               "don't know. If the user asks about his docs use your retrieval tool to search."
           )

           llm_with_tools = llm.bind_tools([retrieve], tool_choice="retrieve")
           prompt = [SystemMessage(system_message_content)] 
           response = llm_with_tools.invoke(prompt + state["messages"])
           # MessagesState appends messages to state instead of overwriting
           return {"messages": [response]}
    
    
       # Step 2: Execute the retrieval.
       tools = ToolNode([retrieve])
    
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
               "answer concise. If the user asks about his docs use your retrieval tool to search."
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
    
           # Run
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
    
       # MEMORY
       # memory = RedisSaver(conn=redisCli)
       memory = MemorySaver()
    
       graph = graph_builder.compile(checkpointer=memory)
    
       config = {"configurable": {"thread_id": "abc123"}}

       loader = PyPDFLoader(tmp_location)
       docs = loader.load()
       print(f"doc length: {len(docs[0].page_content)}.")
       text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
       all_splits = text_splitter.split_documents(docs)
       print(f"split into {len(all_splits)} sub-documents.")
       print (all_splits[0])
       # INDEX CHUNKS
       document_ids = vector_store.add_documents(documents=all_splits)
       print("stored in vector store.")
       print(document_ids)
       st.session_state.loaded = True
       st.session_state.graph = graph
       st.session_state.cff = config
       st.rerun()

else:
    print('duck')
    if "messages" not in st.session_state:
     st.session_state.messages = []
     # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
         st.markdown(message["content"])
     # Create a chat input field to allow the user to enter a message. This will display
     # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):
    # Store and display the current prompt.
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
         st.markdown(prompt)

       for step in st.session_state.graph.stream(
         {"messages": [{"role": "user", "content": prompt}]},
             stream_mode="values",
             config=st.session_state.cff,
       ):
         msg = step["messages"][-1]
         if msg.type == 'ai' and not msg.tool_calls:
            with st.chat_message("assistant"):
                st.write(msg.content)
                st.session_state.messages.append({"role": "assistant", "content": msg.content})