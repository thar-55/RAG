# from langchain_experimental.agents import AgentType
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.prompts import FewShotPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
# langchain_experimental.agents.create_pandas_dataframe_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import streamlit as st
import pandas as pd
import os
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import json
from langchain_google_genai import ChatGoogleGenerativeAI




if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] =st.secrets['GOOGLE_API_KEY']


class Customer(BaseModel):
    name: str
    email: str
    phone: str
    price: str
    quantity: str
    date: str

    def to_dict(self):
        return {
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'price': self.price,
            'quantity': self.quantity,
            'date': self.date,

        }






class Report_Structure(BaseModel):
    customer_list: List[Customer]
    item_title: str
    ref_id: str

    def to_dict(self):
        return {
            'customer_list': [customer.to_dict() for customer in self.customer_list],
            'item_title': self.item_title,
            'ref_id': self.ref_id,

        }


class Refs_Reports(BaseModel):
    reports_list: List[Report_Structure]

    def to_dict(self):
        return {
            'reports_list': [report.to_dict() for report in self.reports_list],

        }





df = pd.read_csv('beta_dataset_v2.csv')


def search_dataframe(user_input):
    """Searches DataFrame for matching ref_id values."""
    if isinstance(user_input, list):  
        user_input = ",".join(user_input)  # Convert list to a string

    search_terms = [term.strip() for term in user_input.replace('\n', ',').replace(' ', ',').split(',') if term]
    st.write(search_terms)

    results = df[df['ref_id'].isin(search_terms)]
    
    report_dict = {}
    for _, row in results.iterrows():
        ref_id = row["ref_id"]
        if ref_id not in report_dict:
            report_dict[ref_id] = {"item_title": row["Item Title"], "customers": []}
        report_dict[ref_id]["customers"].append({
            "name": row["Name"],
            "phone": row["customer_phone"],
            "email": row["customer_email"],
            "quantity": row["Quantity"],
            "price": row["price"],
            'date':row["Date"]
        })
    
    reports_list = [
        {
            "ref_id": ref_id,
            "item_title": details["item_title"],
            "customers": details["customers"]
        }
        for ref_id, details in report_dict.items()
    ]
    return reports_list

# Example Few-Shot Template
examples = [
    {"input": "000", "output": "[{ref_id: '000', item_title: 'Laptop', customers: [{name: 'John Doe', phone: '1234567890', price: '100',email: 'ttt@gmail.com',quantity:'5',date:'3/1/2021'}, {name: 'Bob White', phone: '4321098765', price: '300',email: 'ffffss@gmail.com',quantity:'6',date:'4/1/2023'}]}]"},
    {"input": "222", "output": "[{ref_id: '222', item_title: 'Tablet', customers: [{name: 'Alice Brown', phone: '5678901234', price: '150',email: 'ddaa@gmail.com',quantity:'7',date:'6/1/2025'}]}]"}
]

# examples = [
#     {
#         "input": "000",
#         "output": json.dumps([
#             {
#                 "ref_id": "000",
#                 "item_title": "Laptop",
#                 "customers": [
#                     {"name": "John Doe", "phone": "1234567890", "price": "$100"},
#                     {"name": "Bob White", "phone": "4321098765", "price": "$300"}
#                 ]
#             }
#         ])
#     },
#     {
#         "input": "222",
#         "output": json.dumps([
#             {
#                 "ref_id": "222",
#                 "item_title": "Tablet",
#                 "customers": [
#                     {"name": "Alice Brown", "phone": "5678901234", "price": "$150"}
#                 ]
#             }
#         ])
#     }
# ]
output_parser = JsonOutputParser(
            pydantic_object=Refs_Reports
        )

example_prompt = PromptTemplate(
    input_variables=["input"],
    template=" {input}  {format_instructions}" ,
    partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        } 
)

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="User Input: {input}\nOutput:",
    input_variables=["input"]
)








def generate_pdf(customers, filename="customer_report.pdf"):
    file_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 12)

    c.drawString(50, height - 50, "Customer Report")
    y = height - 80

    for i, customer in enumerate(customers):
        c.drawString(50, y, f"{i + 1}. Name: {customer['name']}")
        c.drawString(50, y - 20, f"   Email: {customer['email']}")
        c.drawString(50, y - 40, f"   Phone: {customer['phone']}")
        c.drawString(50, y - 60, f"   Date: {customer['date']}")
        c.drawString(50, y - 80, f"   Quantity: {customer['quantity']}")
        c.drawString(50, y - 100, f"   Price: {customer['price']}")
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


def format_response(user_input: str):
    """Formats response using LLM."""
    reports = search_dataframe(user_input)
    reports_summary = str(reports)
    st.write(reports)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    # agent = initialize_agent(tools=[], agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, llm=llm)
    formatted_output = llm.invoke(example_prompt.format(input=reports_summary))
    return formatted_output.text



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
    # df = pd.read_csv('beta_dataset_v2.csv')

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


    # llm = ChatOpenAI(
    #     temperature=0, model="gpt-4", openai_api_key=openai_api_key, streaming=False
    # )


    # llm = OpenAI(
    #    model_name="gpt-3.5-turbo-instruct",  temperature=0, openai_api_key=openai_api_key
    # )
    # pandas_df_agent = create_pandas_dataframe_agent(
    # llm,
    # df=[df],
    # verbose=True,
    # allow_dangerous_code=True,
    # agent_type="tool-calling",

    # )
   

    # pandas_df_agent = create_pandas_dataframe_agent(
    #     llm,
    #     df=[df],
    #     allow_dangerous_code=True,
    #     verbose=True,
    #     agent_type="tool-calling",

    # # include_df_in_prompt=True,
    #     # handle_parsing_errors=True,
    # )
    # agent_executor = AgentExecutor(agent=agent, tools=tools)
    # then return  a list of of each ref_id  item title and the list of customers  (name ,customer_phone ,customer_email,date,price , quantity )   thay purchased this item

    with st.chat_message("assistant"):
        # st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        output_parser = JsonOutputParser(
            pydantic_object=Refs_Reports
        )

        # prompt_template = PromptTemplate( template="""using this variable `{input_refs}`.
        # 1.  search in purchases history where the ref_id matches {input_refs}.
        # 2. Return the matching records which will include:
        #    - **Customers who purchased the item** with the corresponding `ref_id`.
        #    - **Item title**.
        #    - **Reference ID** (`ref_id`).
        
        # For the customer data, please include the following information for each customer:
        # - **Name**
        # - **Phone number** (`customer_phone`)
        # - **Email address** (`customer_email`)
        # - **Purchase date**
        # - **Price paid**
        # - **Quantity purchased**
        
        # If no relevant data is found for any of the references, return the message: "error".
        
        # ### Format Instructions:
        # {format_instructions}
        
        # Finally, compile and return a **reports list**. This list will contain individual reports for each `ref_id`, which includes:
        # - **Item title**
        # - **Reference ID** (`ref_id`)
        # - **Customers who purchased the item**

        #             """,
        # input_variables=["input_refs"],
        # partial_variables={
        #     "format_instructions": output_parser.get_format_instructions()
        # }   )
        
        # prompt_template = PromptTemplate( template="""
        # search for records where ref_id={input_refs}
       
        # If no relevant data is found, return: error
        # {format_instructions}
        # reports_list":list of customers and item titles purchased item with ref_id={input_refs}

        # """,
        # input_variables=["input_refs"],
        # partial_variables={
        #     "format_instructions": output_parser.get_format_instructions()
        # }           )
        # prompt_text = prompt_template.format(input_refs=st.session_state.messages)
        # response =pandas_df_agent.invoke
        # response = pandas_df_agent.run(prompt_text, callbacks=[st_cb])
        last_message = st.session_state.messages[-1]["content"]

        st.write(last_message)

        response = format_response(last_message)
        print(response)
        st.write(response)

        try:

            st.write('phase 0')
            formatted_output = output_parser.parse(response)
            new_list = formatted_output['reports_list']
            st.write('phase 1')
            st.write(new_list)
            print("phase 1")
            for i in range(len(new_list)):
                st.session_state["customers"] = new_list[i]['customer_list']
                print("phase 2")
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                print("phase 3")
                if len(new_list[i]['customer_list']) > 0:
                    pdf_file = generate_pdf(new_list[i]['customer_list'])
                    with open(pdf_file, "rb") as f:
                        st.download_button(str(i) + "Download PDF", f, file_name=str(i) + "customer_report.pdf",
                                           mime="application/pdf")

        except Exception as e:
            st.error("No data found.Please Try again." + str(e))


