""" chat example """
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.llms import OpenAI
import streamlit as st
from streamlit_pills import pills
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from chromadb.utils import embedding_functions


# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)


class StreamlitChatPack(BaseLlamaPack):
    """Streamlit chatbot pack."""

    def __init__(
        self,
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if not run_from_main:
            raise ValueError(
                "Please run this llama-pack directly with "
                "`streamlit run [download_dir]/streamlit_chatbot/base.py`"
            )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}


    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""

        st.set_page_config(
            page_title="HistSemanticSearch",
            page_icon="ℹ️",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        # Initialize the chat messages history
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Proszę o pytania o politykę morską Polski w XVI wieku"}
            ]

        st.title(
            "HistSemanticSearch"
        )
        st.info(
            "Serwis odpowiada na pytania związane z polityką morską Polski w XVI wieku - za panowania Zygmunta Augusta",
            icon="ℹ️",
        )

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            )  # Add response to message history

        def load_index_data():

            client = chromadb.PersistentClient(path="../emb")

            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-ada-002"
                )

            collection = client.get_collection(name="bodniak_v3", embedding_function=openai_ef)

            service_context = ServiceContext.from_defaults(
                llm=OpenAI(model="gpt-4", temperature=0.0)
            )

            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                                       service_context=service_context)

            return index


        index = load_index_data()

        selected = pills(
            "Wybierz pytanie lub zadaj swoje własne w polu poniżej.",
            [
                "Czy kaprowie mogli posiadać własne okręty?",
                "Jak były uzbrojone okręty kaprów?",
                "Jaki był udział kaprów w zyskach ze zdobyczy?",
            ],
            clearable=True,
            index=None,
        )

        if "chat_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["chat_engine"] = index.as_chat_engine(
                chat_mode="context",
                verbose=True,
                system_prompt="You are a helpful assistant, a specialist in history",
                similarity_top_k=3
            )

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if selected:
            with st.chat_message("user"):
                st.write(selected)
            with st.chat_message("assistant"):
                response = st.session_state["chat_engine"].stream_chat(selected)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                add_to_message_history("user", selected)
                add_to_message_history("assistant", response)

        if prompt := st.chat_input(
            "Twoje pytanie..."
        ):  # Prompt for user input and save to chat history
            add_to_message_history("user", prompt)

        # If last message is not from assistant, generate a new response
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = st.session_state["chat_engine"].stream_chat(prompt)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                # st.write(response.response)
                add_to_message_history("assistant", response.response)


if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
