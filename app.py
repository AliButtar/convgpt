# Imports
import streamlit as st
import pyperclip

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Title of Web App
st.title("Have better conversations using AI 🧑‍💻💬")

# Ask user to input the text in a text box that dynamically resizes based on the text and add placeholder text
text = st.text_area("Enter the message you want a response to",
                    placeholder="Hi, how are you?", height=200)

col1, col2 = st.columns(2)

with col1:
    # Create dropdown list for user to select kind of response such as happy, sad, angry, etc.
    emotion = st.selectbox("Select the emotion you want to convey", [
        "Happy", "Sad", "Angry", "Neutral"])

with col2:
    # Create dropdown list for user to select the tone of the response such as formal, informal, etc.
    tone = st.selectbox("Select the tone you want to convey",
                        ["Formal", "Informal"])

# Create a text box to take suggestions from the user and add placeholder text
suggestion = st.text_input(
    "Enter your suggestion", placeholder="Make it Brief, Be Polite, 30 Words max, I went hiking etc.")

# Define Templates
convGPT_objective_template = PromptTemplate(
    input_variables=["chat_history", "emotion", "tone", "text", "suggestion"],
    template="""
You are a helpful person assistant that is going to help generate a {emotion} and {tone} reponse to a conversational message. 
Also keep in mind the following suggestions when creating a response to the conversational message: {suggestion}
Conversation History: 

{chat_history}

YOU WILL ACT AS A PERSON AND NOT AN AI
Each reply you make is going to be a direct respone to the following message in the conversation: {text}
"""
)

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        input_key="text", memory_key="chat_history", human_prefix="Message Received", ai_prefix="My Response")


# Initialize the model
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 temperature=0.3)
chain = LLMChain(llm=llm, prompt=convGPT_objective_template,
                 memory=st.session_state.chat_memory, verbose=True)

# Create a button to generate the response
if st.button("Generate Response", use_container_width=True):
    # Create a progress bar to show the progress of generating the response
    with st.spinner("Generating Response..."):
        # Generate the response using the generate_response method
        st.session_state.response = chain.run(text=text, emotion=emotion.lower(
        ), tone=tone.lower(), suggestion=suggestion)

with st.expander("**Use this response**", expanded=True):
    # Write the repsonse in a markdown with green color and create a red outline around the text
    if "response" in st.session_state:
        st.markdown(f":green[{st.session_state.response}]")

        # Create a button to copy the response to clipboard
        if st.button("Copy to Clipboard", use_container_width=True):
            pyperclip.copy(st.session_state.response)
            st.success("Copied to Clipboard")

with st.expander("**See the conversation history**"):
    if "chat_memory" in st.session_state:
        chat_dict = st.session_state.chat_memory.dict()
        message_list = ["### Message Received:\n", "### My Response:\n"]
        for i, message in enumerate(chat_dict["chat_memory"]["messages"]):
            if i % 2 == 0:
                st.write(message_list[0])
            else:
                st.write(message_list[1])
            st.write(message["content"])

# streamlit run app.py --runner.fastReruns=False --server.runOnSave=False --server.port=8501 --server.headless=True --global.developmentMode=False
