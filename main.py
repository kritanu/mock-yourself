from util import *
from streamlit_option_menu import option_menu
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from gtts import gTTS
from io import BytesIO


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Mock Yourself", page_icon=":robot_face:", layout="centered")

# --- SETUP SESSION STATE VARIABLES ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = False
if "response" not in st.session_state:
    st.session_state.response = None
if "prompt_activation" not in st.session_state:
    st.session_state.prompt_activation = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "prompt" not in st.session_state:
    st.session_state.prompt = False

load_dotenv()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header('Configuration')
groq_api_key = sidebar_api_key_configuration()
model = sidebar_groq_model_selection()

# --- MAIN PAGE CONFIGURATION ---
st.title("Mock Yourself :books:")
st.write("A personal interviewer who mocks you based on your resume!")
st.write(':blue[***Powered by Groq AI, Langchain, Streamlit.***]')

# ---- NAVIGATION MENU -----
selected = option_menu(
    menu_title=None,
    options=["Mock Yourself", "Reference", "About"],
    icons=["robot", "bi-file-text-fill", "app"],  # https://icons.getbootstrap.com
    orientation="horizontal",
)

llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)

prompt = ChatPromptTemplate.from_template(
    """
    You are an interviewer. You are hiring for a {role} role in your company. Ask a total of {number_q} questions of {difficulty} difficulty based on the provided resume context only. Few of them could be behavioural questions if ```{behaviour}``` is True. If the user asks some extra question, please answer them based on the context. If the question is out of context, do not try to answer and respond that the asked question is out of context or something similar.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    Questions: {input}
    """
)
# ----- SETUP Doc Chat MENU ------
if selected == "Mock Yourself":

    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("Desired Role", placeholder="Software Engineer")
        number_q = st.number_input("Number of Questions", value="min", placeholder="Type a number...", min_value=1, max_value=20)

    with col2:
        difficulty = st.radio("Difficulty",["Easy", "Medium", "Hard"], index=1)
        behaviour = st.checkbox("Behavioural Questions")

    st.subheader("Upload Resume")
    pdf_docs = st.file_uploader("Upload your resume", type=['pdf'], accept_multiple_files=True,
                                disabled=not st.session_state.prompt_activation, label_visibility='collapsed')
    process = st.button("Process", type="primary", key="process", disabled=not pdf_docs)

    if process:
        with st.spinner("Processing ..."):
            st.session_state.vector_store = create_vectorstore(pdf_docs)
            st.session_state.prompt = True
            st.success('Database is ready')

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Are you ready?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    container = st.container(border=True)
    if question := st.chat_input(placeholder='Enter your question related to uploaded document',
                                 disabled=not st.session_state.prompt):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner('Processing...'):
            st.session_state.response = get_llm_response(llm, prompt, question,behaviour, role, number_q, difficulty)

            sound_file = BytesIO()
            tts = gTTS(st.session_state.response['answer'], lang='en')
            tts.write_to_fp(sound_file)
    
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.response['answer']})
            st.chat_message("assistant").write(st.session_state.response['answer'])
            st.audio(sound_file)

# ----- SETUP REFERENCE MENU ------
if selected == "Reference":
    st.title("Reference & Context")
    if st.session_state.response is not None:
        for i, doc in enumerate(st.session_state.response["context"]):
            with st.expander(f'Reference # {i + 1}'):
                st.write(doc.page_content)

# ----- SETUP ABOUT MENU ------
if selected == "About":
    with st.expander("About this App"):
        st.markdown(''' This app crafts questions for you based on your resume.

    - Allows to chat with your own resume.
    - Displays the response context and document reference.

        ''')
    with st.expander("Which Large Language models are supported by this App?"):
        st.markdown(''' This app supports the following LLMs as supported by Groq:

    - Chat Models -- Groq
        - Llama3-8b-8192 
        - Llama3-70b-8192 
        - Mixtral-8x7b-32768
        - Gemma-7b-it
        ''')

    with st.expander("Which library is used for vectorstore?"):
        st.markdown(''' This app supports the FAISS for AI similarity search and vectorstore:
        ''')
