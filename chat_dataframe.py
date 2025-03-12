# from langchain_experimental.agents import AgentType
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
# langchain_experimental.agents.create_pandas_dataframe_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
import os
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser

class Report_Structure(BaseModel):
    customer_list: List[str]


    def to_dict(self):
        return {
            'customer_list': [str(customer) for customer in self.customer_list],
     

        }

def generate_prompt():
    prompt = (
        """ Generate {questions_number} multiple choice questions for the subject {subject} with difficulty level {difficulty}.

        The questions should cover the following topics and sub-topics:
        {topics_str}
        
        
        For each question, provide:
        1. The question text.
        2. A list of multiple choice options.
        3. The index of the correct answer.
        4. A short explanation of why the correct answer is correct and why distractor choice is wroung.
        5. A list of related topics and subtopics relevant to the question.

        {format_instructions}
        questions_list":list of each question (question,choices,answer,explanation,related_topics)
        answers:  list of correct answers index in each question
        
         """
    )






openai_api_key = st.secrets['OPENAI_API_KEY']
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain: Chat with pandas DataFrame")

uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

if not uploaded_file:
    st.warning(
        "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )

if uploaded_file:
    df = load_data(uploaded_file)

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0, model="gpt-4o", openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        allow_dangerous_code=True ,
        # verbose=True,
        agent_type="tool-calling",
        # handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        output_parser = JsonOutputParser(
        pydantic_object=Report_Structure
        )
        prompt_template = PromptTemplate( template="""search in purchases history for the ref_id = {input_ref} , then return  a list of customer names thar purchased this item 
           {format_instructions}
        customer_list":list of customers (name)
        
        """,
        input_variables=["input_ref"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        }           )
        prompt_text = prompt_template.format(input_ref=st.session_state.messages)
        # response =pandas_df_agent.invoke
        response = pandas_df_agent.run(prompt_texts, callbacks=[st_cb])
        print(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
