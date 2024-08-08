from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_milvus import Milvus
import os
import boto3
from pymilvus import MilvusClient
import tempfile
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_experimental.text_splitter import SemanticChunker
import botocore

session = boto3.Session()

config = botocore.config.Config(
    region_name=session.region_name,
    signature_version='v4',
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

bedrock_client = session.client('bedrock-runtime', config=config)
s3_client = session.client('s3', config=config)

bedrock_llm = ChatBedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_client,
    model_kwargs={"temperature": 0}
)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)
embeddings = bedrock_embeddings

print('start')
class CustomS3Loader:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')

    def download_file(self, s3_key, local_path):
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)

    def load(self):
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            for obj in response.get('Contents', []):
                s3_key = obj['Key']
                local_path = os.path.join(temp_dir, s3_key)
                self.download_file(s3_key, local_path)

                if local_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(local_path)
                    documents.extend(loader.load())
                elif local_path.endswith(".docx"):
                    loader = Docx2txtLoader(local_path)
                    documents.extend(loader.load())
        return documents

# Load documents from the "doc-listener" S3 bucket
print("loading docs")
loader = CustomS3Loader(bucket_name="fatima-rag")
documents = loader.load()
print("Number of documents loaded:", len(documents))
print("First document content:", documents[0].page_content[:100] if documents else "No documents loaded")
semantic_chunker = SemanticChunker(bedrock_embeddings)
splits = semantic_chunker.create_documents([d.page_content for d in documents])
print("Number of splits:", len(splits))
print("Empty splits:", [i for i, s in enumerate(splits) if not s.page_content.strip()])
splits = [s for s in splits if s.page_content.strip()]
client = MilvusClient("milvus_demo.db")

print('initializing vectorstore')

# Store document embeddings in Milvus
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=bedrock_embeddings,
    connection_args={"uri": "./milvus_demo.db"},
    drop_old=False,
)

print("vectorstore complete")
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

system_prompt = """
You are an AI assistant and provide answers to questions using fact-based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If the question is unrelated to the context enclosed in <context> tags then please give it a small reply without trying  to mention anything from the context.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
<context>
{context}
</context>
<question>
{question}
</question>
The response should be specific and use statistics or numbers when possible.CONTEXT INFORMATION:
{context}

QUESTION: {question}

ANSWER: Provide a detailed answer
CITATIONS: After completing your answer, under a 'Sources' heading, list the most relevant two to three lines to your answer from the context word to word, donot summarize or explain.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history,
formulate a standalone question that can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    bedrock_llm,
    retriever,
    contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(
    llm=bedrock_llm,
    prompt=prompt,
    document_variable_name="context"
)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []
question = 'why should i work at emumba'


response = rag_chain.invoke({
        "chat_history": chat_history,  
        "input": question,
        "question": question
    })
    

print("Answer:", response['answer'])
    
chat_history.append({'type': 'user', 'content': question})
chat_history.append({'type': 'ai', 'content': response['answer']})
