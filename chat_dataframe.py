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
from typing import List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

df = pd.read_csv('beta_dataset.csv')


class Customer(BaseModel):
    name: str
    email: str
    phone: str


    def to_dict(self):
        return {
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            
        }


class Report_Structure(BaseModel):
    customer_list: List[Customer]


    def to_dict(self):
        return {
            'customer_list': [customer.to_dict() for customer in self.customer_list],
     

        }

def generate_pdf(customers, filename="customer_report.pdf"):
    file_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 12)
    
    c.drawString(50, height - 50, "Customer Report")
    y = height - 80
    
    for i, customer in enumerate(customers):
        c.drawString(50, y, f"{i+1}. Name: {customer['name']}")
        c.drawString(50, y - 20, f"   Email: {customer['email']}")
        c.drawString(50, y - 40, f"   Phone: {customer['phone']}")
        y -= 70
        if y < 50:  # Add a new page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
    
    c.save()
    return file_path







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


st.set_page_config(page_title="Customers report builder", page_icon="🦜")
st.title("🦜 Generate your report with AI ")

# uploaded_file = st.file_uploader(
#     "Upload a Data file",
#     type=list(file_formats.keys()),
#     help="Various File formats are Support",
#     on_change=clear_submit,
# )

# if not uploaded_file:
#     st.warning(
#         "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
#     )

# if uploaded_file:
#     df = load_data(uploaded_file)

# if "customers" in st.session_state and st.session_state["customers"]:
#     if st.button("Generate PDF"):
#         pdf_file = generate_pdf(st.session_state["customers"])
#         with open(pdf_file, "rb") as f:
#             st.download_button("Download PDF", f, file_name="customer_report.pdf", mime="application/pdf")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Enter the reference number "):
    

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
        prompt_template = PromptTemplate( template="""search in purchases history for the ref_id = {input_ref} , then return  a list of customers  (name ,customer_phone ,customer_email)   thay purchased this item 
          
           If no relevant data is found, return: error}
           {format_instructions}
        customer_list":list of customers 
        
        """,
        input_variables=["input_ref"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        }           )
        prompt_text = prompt_template.format(input_ref=st.session_state.messages)
        # response =pandas_df_agent.invoke
        response = pandas_df_agent.run(prompt_text, callbacks=[st_cb])
        print(response)

        try:
            data = json.loads(response)
            if "error" in data:
                st.warning("No data found. Try again with a different request.")
            else:
                formatted_output = output_parser.parse(response)
                st.session_state["customers"] = formatted_output['customer_list']
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                pdf_file = generate_pdf(formatted_output['customer_list'])
                with open(pdf_file, "rb") as f:
                    st.download_button("Download PDF", f, file_name="customer_report.pdf", mime="application/pdf")
        except json.JSONDecodeError:
            st.error("Failed to parse response. Try again.")
        
