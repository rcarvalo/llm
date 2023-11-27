import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.document_loaders.csv_loader import CSVLoader  # using CSV loaders
from langchain.document_loaders import DirectoryLoader

import csv
import tempfile 
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
DEVICE = 'mps'
DB_FAISS_PATH = '.'

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
# Define a dictionary to map file extensions to their respective loaders
loaders = {
    '.pdf': PyMuPDFLoader,
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
    '.txt': TextLoader,
}

# Define a function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )





def evaluate(csv_file,qa_chain,model_name,parameters):
    
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        with open(model_name+'.csv', mode='w') as result_file:
            result_writer = csv.writer(result_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Question', 'ResultLLM', 'Baseline'])
            for row1,row2 in csv_reader:
                print('row : ', str(row1))
                result_writer.writerow([row1,qa_chain({'question':row1, 'chat_history':[]})['answer'],row2])
def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Models from Hugging Face] LLM model

    ''')
    add_vertical_space(5)
    st.write('Made by [Rémi Carvalot]')

load_dotenv()

def main():
    st.header("Chat with pdf")

    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='csv')
    
    #st.write(pdf)
    if pdf is not None:
        #pdf reader
        

        '''
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        '''
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf.getvalue())
            tmp_file_path = tmp_file.name
        print(pdf.name)
        #loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8",csv_args={'delimiter': ','}) # any loader can be put here based on the data being used
        # Create DirectoryLoader instances for each file type
        txt_loader = create_directory_loader('.txt', './txt_perfect')
        #pdf_loader = create_directory_loader('.pdf', './files')
        #xml_loader = create_directory_loader('.xml', './files')
        #csv_loader = create_directory_loader('.csv', './files')

        # Load the files
        txt_documents = txt_loader.load()
        #csv_documents = csv_loader.load()
        
        
        #loader = DirectoryLoader('./files')#,glob="**/*.md", show_progress=True)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64,
            #length_function=len
            )
        '''
        csv_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            length_function=len
            )'''
        #split documents
        chunks = text_splitter.split_documents(txt_documents)
        #chunks.extend(csv_splitter.split_documents(csv_documents))
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = HuggingFaceInstructEmbeddings(
                #model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}$
                model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": DEVICE}
            )
            #embeddings = OpenAIEmbeddings()
            #from_texts
            VectorStore = FAISS.from_documents(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        #VectorStore.save_local(DB_FAISS_PATH)
        # embeddings = OpenAIEmbeddings()
        #VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            model_name = 'dolphin-2.0-mistral-7b'
            parameters = {'chunk_size':'1000',
              'chunk_overlap':'200',
              'model':'dolphin-2.0',
              'quantization':'GGUF',
              'preprocessing':'Directory',
              'embedding':'hkunlp/instructor-large',
              'device':'MPS',
              'chunk_technique':'RecursiveTextSplitter'
              }
            #docs = VectorStore.similarity_search(query=query, k=3)
            llm = CTransformers(
                #model="TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
                #model_file="./dolphin-2.2.1-mistral-7b.Q4_K_M.gguf",
                model='dolphin-2.0-mistral-7b.Q4_K_M.gguf',
                model_type='mistral',
                #model="TheBloke/Llama-2-13B-chat-GGUF",
                #model_file="llama-2-13b-chat.Q4_0.gguf",
                config={"max_new_tokens": 300, "context_length": 2048, "temperature": 0, "gpu_layers":1},
            )
            #llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

            template = generate_prompt(
                """
            {context}

            Question: {question}
            """,
                system_prompt=SYSTEM_PROMPT,
            )
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                #chain_type="stuff",
                retriever=VectorStore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                #chain_type_kwargs={"prompt": prompt},
            )
            #chain = ConversationalRetrievalChain.from_llm(llm, retriever=VectorStore.as_retriever())
            #response = chain({'question':query, 'chat_history':[]})
            evaluate('questionDataset.csv',chain,model_name,parameters)
            with open('parameter'+model_name+'.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in parameters.items():
                    writer.writerow([key, value])
            #st.write(response)
 
if __name__ == '__main__':
    main()

