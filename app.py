
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
import streamlit as st

from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from openai import OpenAI
import os
from PyPDF2 import PdfReader
from agents import Runner, trace
import asyncio

import numpy as np
import sounddevice as sd
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
set_default_openai_key(OPENAI_API_KEY)

# --- Agent: Search Agent ---
search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)


client = OpenAI(api_key=OPENAI_API_KEY)

def upload_file(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}
    
vector_store_detail = create_vector_store("ACME Shop Product Knowledge Base")


# --- Agent: Knowledge Agent ---
knowledge_agent = Agent(
    name="KnowledgeAgent",
    instructions=(
        "You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool."
    ),
    tools=[FileSearchTool(
            max_num_results=3,
            vector_store_ids=[vector_store_detail["id"]],
        ),],
)

# --- Tool 1: Fetch account information (dummy) ---
@function_tool
def get_account_info(user_id: str) -> dict:
    """Return dummy account info for a given user."""
    return {
        "user_id": user_id,
        "name": "Bugs Bunny",
        "account_balance": "£72.50",
        "membership_status": "Gold Executive"
    }

# --- Agent: Account Agent ---
account_agent = Agent(
    name="AccountAgent",
    instructions=(
        "You provide account information based on a user ID using the get_account_info tool."
    ),
    tools=[get_account_info],
)

# --- Agent: Triage Agent ---
triage_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountAgent for account-related queries
- KnowledgeAgent for questions about Charles
- SearchAgent for anything requiring real-time web search
"""),
    handoffs=[account_agent, knowledge_agent, search_agent],
)

async def test_queries():


    examples = [
        "What's my ACME account balance doc? My user ID is 1234567890", # Account Agent test
        "Tell me about Charles", # Knowledge Agent test
        "Hmmm, what about duck hunting gear - what's trending right now?", # Search Agent test

    ]
    st.write("---")
    st.header("Different agents handling different queries")
    with trace("Maste Agent App Assistant"):
        for query in examples:
            result = await Runner.run(triage_agent, query)
            st.write(f"User: {query}")
            st.write(f"Agent: {result.final_output}") 
            st.write("---")

async def voice_assistant():
    samplerate = sd.query_devices(kind='input')['default_samplerate']

    with st.spinner("Initializing voice assistant..."):
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_agent))

        # Check for input to either provide voice or exit
        cmd = st.text_input("Press Enter to speak your query (or type 'esc' to exit): ", key="input for audio")
        if cmd.lower() == "esc":
            st.write("Exiting...")
            return
        else:
            st.write("Listening...")

            st.title("Record Audio Locally")

            # Record audio
            duration = st.slider("Select recording duration (seconds)", 1, 10, 5)
            if st.button("Start Recording"):
                st.write("Recording...")
                samplerate = 44100  # Sample rate
                recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
                sd.wait()  # Wait until recording is finished
                st.success("Recording finished!")

                # Input the buffer and await the result
                audio_input = AudioInput(buffer=recording)

                with trace("ACME App Voice Assistant"):
                    result = pipeline.run(audio_input)

                # Transfer the streamed result into chunks of audio
                response_chunks = []
                async for event in result.stream():
                    if event.type == "voice_stream_event_audio":
                        response_chunks.append(event.data)

                response_audio = np.concatenate(response_chunks, axis=0)

                # Play response
                st.write("Assistant is responding...")
                sd.play(response_audio, samplerate=samplerate)
                sd.wait()
                print("---")


async def main():
    st.header("Build vector store and upload files")
    pdf_docs = st.file_uploader("Upload your knowledge base document", type=["pdf"], accept_multiple_files=False)
    if st.button("Submit & Process"):
        with st.spinner("Processing your PDF documents..."):
            if pdf_docs:
                save_path = f"./{pdf_docs.name}"
                    # Save the file locally
                with open(save_path, "wb") as f:
                    f.write(pdf_docs.getbuffer())
                st.success("File saved successfully")
                upload_file(save_path, vector_store_detail["id"])
                st.success("Vector store processed successfully")

        st.header("Let the agents do the work")
        st.markdown("""
            You are the virtual assistant Welcome the user and ask how you can help. 
            Based on the user's intent, route to:
            - AccountAgent for account-related queries
            - KnowledgeAgent for questions about Charles
            - SearchAgent for anything requiring real-time web search""")

        st.markdown(
            """ 
            ### Test the agents with some example queries:
            - "What's my ACME account balance doc? My user ID is 1234567890"
            - "Tell me about Charles"
            - "Hmmm, what about duck hunting gear - what's trending right now?"
            """
        )
        await test_queries()

        st.header("Voice Assistant")
        st.markdown(
            """
            ### Speak your query to the assistant:
            - Press Enter to start listening
            - Type 'esc' to exit
            """
        )
        await voice_assistant()

if __name__ == "__main__":
    asyncio.run(main())
