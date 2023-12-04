from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.document_loaders.csv_loader import CSVLoader  # using CSV loaders
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.document_transformers import (
    LongContextReorder,
)
import time
from langchain.prompts import PromptTemplate
import csv
csv_file = 'questionDataset.csv'
DB_FAISS_PATH = '.'
DEVICE = 'mps'
import uuid

'''
parameters = {'chunk_size':1000,
        'chunk_overlap':200,
        'model_name':'starling-lm-7b-alpha.Q4_K_M.gguf',#'dolphin-2.0-mistral-7b.Q4_K_M.gguf',#'dolphin-2.0-mistral-7b.Q4_K_M.gguf',
        'model_type':'mistral',
        'quantization':'GGUF',
        'preprocessing':'PaddleOCR',
        'embedding_model_name':"BAAI/bge-large-en-v1.5",#'hkunlp/instructor-large',
        'device':'MPS',
        'chunk_technique':'RecursiveTextSplitter',
        'max_new_tokens':200,
        'context_length':2048,
        'temperature':0.10,
        'gpu_layers':20
        }
'''
parameters=  {'model_name':'llama-2-13b-chat.Q4_K_M.gguf',#'openchat_3.5.Q4_K_M.gguf',#'starling-lm-7b-alpha.Q4_K_M.gguf',
 'model_type':'llama',
 'method_preprocessing':'PaddleOCR',
 'embedding_model':'BAAI/bge-large-en-v1.5',
 'chunk_technique':'RecursiveTextSplitter',
 'chunk_size':200,
 'chunk_overlap':64,
 'context_length':2048,
 'size_model':'7B',
 'size_answer':200,
 'temperature':0.10,
 'gpu_layers':20,
 'search_kwargs':5,
 'reorder':True,
 'quantization':'GGUF',
 'device':'MPS',
 'rate':0,
 'rate_table':0,
 'rate_text':0}
# Define a dictionary to map file extensions to their respective loaders
loaders = {
    '.csv': CSVLoader,
    '.txt': TextLoader,
}

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()

# Define a function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )

def load(parameters):
    txt_loader = create_directory_loader('.txt', './txt_perfect')
    csv_loader = create_directory_loader('.csv', './csv')

    # Load the files
    txt_documents = txt_loader.load()
    csv_documents = csv_loader.load()
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=64,
        length_function=len
        )

    csv_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=64,
        length_function=len
        )
    ## split documents
    chunks = text_splitter.split_documents(txt_documents)
    chunks.extend(csv_splitter.split_documents(csv_documents))
    ## embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=parameters['embedding_model'], model_kwargs={"device": DEVICE}
    )
    return chunks,embeddings

def result(parameters,chunks,embeddings,query):

   
    retriever = FAISS.from_documents(chunks, embedding=embeddings).as_retriever(
                    search_kwargs={"k": 5}
                )
    llm = CTransformers(
        model=parameters['model_name'],#'TheBloke/WizardLM-7B-uncensored-GGML',#'dolphin-2.0-mistral-7b.Q4_K_M.gguf',
        model_type=parameters['model_type'],
        config={"max_new_tokens": parameters['size_answer'], "context_length": parameters['context_length'], "temperature": parameters['temperature'], "gpu_layers":parameters['gpu_layers']},
    )
    #chain = load_qa_chain(llm=llm, chain_type="stuff")
    docs = retriever.get_relevant_documents(query)
    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end.  If you don't know the answer, just say that you don't know, don't try to make up an answer."
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    template = generate_prompt(
        """
    {context}

    Question: {query}
    """,
        system_prompt=SYSTEM_PROMPT,
    )
    
    promptStr = """GPT4 User: Use the following pieces of context to answer the question at the end.  If you don't know the answer, just say that you don't know, don't try to make up an answer. {context} Please answer the following question: {query}<|end_of_turn|>GPT4 Assistant:"""

    prompt = PromptTemplate(
        template=template, input_variables=["context", "query"]
    )


    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    document_variable_name="context"
    # Instantiate the chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    
    response = chain.run(input_documents=reordered_docs, query=query)
    return response

def create_parameters_file(parameters,model_name,id):
    with open('./results/parameter'+model_name+str(id)+'.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([key for key, value in parameters.items()])
        writer.writerow([value for key, value in parameters.items()])

        
def evaluate(csv_file,model_name,parameters):
    chunks,embeddings = load(parameters)
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        with open('./results/'+model_name+'.csv', mode='w') as result_file:
            result_writer = csv.writer(result_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Question', 'ResultLLM', 'Baseline'])
            
            for row1,row2 in csv_reader:
                print('row : ', str(row1))
                response = result(parameters,chunks,embeddings,row1)
                result_writer.writerow([row1,response,row2])
    create_parameters_file(parameters,model_name,uuid.uuid4())

start = time.time()
evaluate(csv_file,parameters['model_name'],parameters)
end = time.time()
print(end - start)