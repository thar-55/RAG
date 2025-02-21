# **Step-by-Step Implementation of RAG with PDFs (Text, Images, and Tables) Using AWS Bedrock, Pinecone, and LangChain**

This guide will implement **Retrieval-Augmented Generation (RAG)** by:
- **Extracting text, images, and tables** from PDFs.
- **Generating embeddings for each type of data** (Text, Image, Table) using **AWS Bedrock Titan Embeddings**.
- **Storing embeddings in Pinecone** for retrieval.
- **Using AWS Bedrock LLMs** to generate answers from the retrieved context.

---

## **🟢 Step 1: Install Required Libraries**
First, install all the required dependencies:
```bash
pip install langchain langchain-community boto3 pinecone-client pypdf pdf2image pytesseract pdfplumber numpy openai
```
- **`langchain`** → Manages document processing and retrieval.
- **`boto3`** → AWS SDK to interact with Bedrock.
- **`pinecone-client`** → Pinecone vector database for storage.
- **`pypdf`** → Extracts text from PDFs.
- **`pdf2image` & `pytesseract`** → Extracts text from images.
- **`pdfplumber`** → Extracts tabular data.
- **`numpy`** → Handles embeddings.

---

## **🟢 Step 2: Configure AWS Bedrock & Pinecone**
### **2.1 Configure AWS Bedrock**
1. **Create an IAM user** with **Bedrock Full Access** and **S3 Access**.
2. **Set up AWS CLI for authentication**:
   ```bash
   aws configure
   ```
   Enter:
   - AWS **Access Key**
   - AWS **Secret Key**
   - AWS **Region** (e.g., `us-east-1`)

### **2.2 Set Up Pinecone**
1. **Sign up for Pinecone** → [Pinecone.io](https://www.pinecone.io/)
2. **Create a Pinecone Index**:
   - **Dimension:** `1536` (Titan Embeddings)
   - **Metric:** `cosine`
3. **Store Pinecone API Key** as an environment variable:
   ```bash
   export PINECONE_API_KEY="your-pinecone-api-key"
   ```

---

## **🟢 Step 3: Extract Text, Images, and Tables from PDF**
Since PDFs contain **text, images, and tables**, we extract them separately.

### **3.1 Extract Text Using `pypdf`**
```python
import pypdf

def extract_text_from_pdf(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

pdf_text = extract_text_from_pdf("sample.pdf")
print("Extracted Text:", pdf_text)
```

---

### **3.2 Extract Images Using `pdf2image` & `pytesseract`**
```python
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path)
    extracted_text = ""

    for img in images:
        text = pytesseract.image_to_string(img)
        extracted_text += text + "\n"

    return extracted_text

image_text = extract_text_from_images("sample.pdf")
print("Extracted Image Text:", image_text)
```

---

### **3.3 Extract Tables Using `pdfplumber`**
```python
import pdfplumber

def extract_tables_from_pdf(pdf_path):
    extracted_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join([" | ".join(row) for row in table if row])
                extracted_tables.append(table_text)
    return extracted_tables

pdf_tables = extract_tables_from_pdf("sample.pdf")
print("Extracted Tables:", pdf_tables)
```

---

## **🟢 Step 4: Generate Embeddings Using AWS Bedrock**
Now, we convert **text, image text, and table text** into **vector embeddings**.

```python
import boto3
import json
import numpy as np

# Initialize AWS Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_bedrock_embedding(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"text": text})
    )
    embedding = json.loads(response["body"].read())["embedding"]
    return np.array(embedding)

# Generate embeddings for extracted data
text_embedding = get_bedrock_embedding(pdf_text)
image_embedding = get_bedrock_embedding(image_text)
table_embedding = get_bedrock_embedding("\n".join(pdf_tables))

print("Embeddings Generated Successfully!")
```

---

## **🟢 Step 5: Store Embeddings in Pinecone**
### **5.1 Initialize Pinecone**
```python
import pinecone

pinecone.init(api_key="your-pinecone-api-key", environment="us-east-1")

# Define Pinecone index
index_name = "rag-pdf-index"
index = pinecone.Index(index_name)
```

---

### **5.2 Store Embeddings (Text, Image, Table) in Pinecone**
```python
import uuid

def store_in_pinecone(data_type, text, embedding):
    doc_id = str(uuid.uuid4())  # Unique ID
    metadata = {"type": data_type, "text": text}
    index.upsert(vectors=[(doc_id, embedding.tolist(), metadata)])

# Store text, image text, and table embeddings
store_in_pinecone("text", pdf_text, text_embedding)
store_in_pinecone("image", image_text, image_embedding)
store_in_pinecone("table", "\n".join(pdf_tables), table_embedding)

print("Data Stored in Pinecone!")
```
✅ **All extracted data (Text, Images, Tables) are now stored in Pinecone!**

---

## **🟢 Step 6: Implement RAG (Retrieval-Augmented Generation)**
Now, let’s **retrieve relevant embeddings** and use **AWS Bedrock LLM** to generate answers.

### **6.1 Retrieve Similar Documents from Pinecone**
```python
def retrieve_from_pinecone(query):
    query_embedding = get_bedrock_embedding(query)
    results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    
    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]
    return " ".join(retrieved_texts)

query = "What are the key points in the document?"
retrieved_context = retrieve_from_pinecone(query)
print("Retrieved Context:", retrieved_context)
```

---

### **6.2 Generate Response Using AWS Bedrock LLM**
```python
def generate_response_with_bedrock(query, context):
    prompt = f"Using the following document, answer the query:\n\nDocument: {context}\n\nQuery: {query}\nAnswer:"
    
    response = bedrock.invoke_model(
        modelId="anthropic.claude-v2",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"prompt": prompt, "max_tokens": 200})
    )
    
    result = json.loads(response["body"].read())["completion"]
    return result

answer = generate_response_with_bedrock(query, retrieved_context)
print("Generated Answer:", answer)
```

✅ **Now, AWS Bedrock LLM generates an answer based on retrieved documents!**

---

## **🟢 Step 7: Deploy RAG as an API**
To expose this as an API, use Flask.

### **7.1 Install Flask**
```bash
pip install flask
```

### **7.2 Create `app.py`**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag_query():
    query = request.json["query"]
    retrieved_context = retrieve_from_pinecone(query)
    answer = generate_response_with_bedrock(query, retrieved_context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
```

✅ **Now you have a fully functional RAG-based chatbot API!**

---

## **🚀 Summary**
| **Step** | **Task** |
|----------|---------|
| **1** | Install dependencies |
| **2** | Configure AWS Bedrock & Pinecone |
| **3** | Extract text, images, and tables from PDFs |
| **4** | Generate embeddings with Bedrock |
| **5** | Store embeddings in Pinecone |
| **6** | Retrieve relevant context & generate responses using LLM |
| **7** | Deploy as an API |

🚀 **Now you have a complete RAG pipeline! Would you like a guide on deploying this to AWS Lambda/API Gateway?**

It looks like you want to follow the **same RAG implementation** using **AWS Bedrock, Pinecone, and LangChain**, but also need a guide to **deploy it to AWS Lambda and API Gateway**.

---

# **🟢 Deploy RAG-Based Chatbot to AWS Lambda & API Gateway**
This guide explains how to **convert your Flask API into a serverless AWS Lambda function**, making it accessible via **API Gateway**.

---

## **🟢 Step 1: Install AWS CDK and Serverless Framework**
First, install the required AWS deployment tools.

```bash
pip install aws-cdk serverless awscli boto3
```
- **AWS CDK (Cloud Development Kit)** → Automates infrastructure setup.
- **Serverless Framework** → Simplifies Lambda deployment.

---

## **🟢 Step 2: Modify the API Code for AWS Lambda**
AWS Lambda requires a **handler function** instead of Flask routes.

### **2.1 Install `aws-lambda-adapter`**
```bash
pip install aws-lambda-adapter
```

### **2.2 Update `app.py` for AWS Lambda**
```python
from flask import Flask, request, jsonify
from aws_lambda_wsgi import response as lambda_response

app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag_query():
    query = request.json["query"]
    retrieved_context = retrieve_from_pinecone(query)
    answer = generate_response_with_bedrock(query, retrieved_context)
    return jsonify({"answer": answer})

# Lambda Handler
def lambda_handler(event, context):
    return lambda_response(app, event, context)

if __name__ == "__main__":
    app.run(debug=True)
```
✅ **Now, this code is compatible with AWS Lambda!**

---

## **🟢 Step 3: Create a Serverless Configuration (`serverless.yml`)**
This file automates **Lambda & API Gateway deployment**.

### **3.1 Create `serverless.yml`**
```yaml
service: rag-chatbot

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  timeout: 30

functions:
  rag-api:
    handler: app.lambda_handler
    events:
      - http:
          path: rag
          method: post
```

---

## **🟢 Step 4: Deploy the API to AWS Lambda**
Run the following command:
```bash
serverless deploy
```
✅ **This will:**
1. **Package the Python application.**
2. **Deploy it as a Lambda function.**
3. **Expose it via API Gateway.**

---

## **🟢 Step 5: Test the API**
After deployment, Serverless will return an **API Gateway URL**.

### **5.1 Call API via `curl`**
```bash
curl -X POST "https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key points in the document?"}'
```

✅ **You now have a fully deployed RAG-based chatbot on AWS!**

---

## **🟢 Summary**
| **Step** | **Task** |
|----------|---------|
| **1** | Install AWS CDK & Serverless |
| **2** | Modify Flask App for Lambda |
| **3** | Configure `serverless.yml` |
| **4** | Deploy to AWS Lambda & API Gateway |
| **5** | Test API using `curl` |

🚀 **Your RAG chatbot is now running serverlessly! Would you like a guide on adding authentication (e.g., AWS Cognito) for secure access?**