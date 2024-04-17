import os
import gc 
import chromadb
import PyPDF2
import time
import accelerate
import torch
# from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFacePipeline
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline,BitsAndBytesConfig,GenerationConfig


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark=True
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16 "
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
# os.system('export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=16"')


def pdf_to_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text


# Initialize text splitter and embeddings
# c_s=5000,c_o=500

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# Convert PDF to text
text = pdf_to_text(os.path.join('./', '../2024.pdf'))
docs = [Document(page_content=x) for x in text_splitter.split_text(text)] #very important converting str to documents
# texts = text_splitter.split_documents(docs)
texts= docs
# print(texts[:3])

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
# BAAI/bge-large-en-v1.5
print("============================pdf_content converted to docs=================================")

persist_directory = 'db'
## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large",cache_folder="./")

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
print("============================db is created=================================")

# persiste the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)



#zephyr-7b-->HuggingFaceH4/zephyr-7b-beta
# NousResearch/Hermes-2-Pro-Mistral-7B
#Xwin-LM/Xwin-LM-13B-V0.1
# Xwin-LM/Xwin-LM-7B-V0.2
# snorkelai/Snorkel-Mistral-PairRM-DPO
# quantization_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True,
#                                          llm_int8_threshold=200.0)
# quantization_config=quantization_config --> inside AutoModelForCausalLM
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # load_in_8bit_fp32_cpu_offload=True,

        # bnb_4bit_compute_dtype="torch.bfloat16",
)

tokenizer = AutoTokenizer.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.1", cache_dir="../")

# load_in_4bit=True,device_map="auto" ,trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.1", cache_dir="../",device_map="auto",
                                             quantization_config=quantization_config,
                                             low_cpu_mem_usage=True, pad_token_id=0,resume_download=True,do_sample=True)
# generate_config=GenerationConfig.from_pretrained("zephyr-7b-->HuggingFaceH4/zephyr-7b-beta",cache_dir="../")

generate_text = pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation', #question-ans,text2text,image2text 
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=1024,  # max number of tokens to generate in the output
    # max_length=6000,
    # max_length=512,
    # min_length=100,
    # generation_config=generate_config,
    batch_size=4,
    repetition_penalty=1.15  # without this output begins repeating
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=generate_text,
    model_kwargs={"temperature": 0.1, "max_length": 512},
)

print("============================retriever is created=================================")
# search_kwargs={"k": 2}
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

########################################################

# prompt_template = """So, your question is:
# {out}

# Question: {question}
# """
# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# chain_type_kwargs = {"prompt": PROMPT}
# ,chain_type_kwargs=chain_type_kwargs

qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever,return_source_documents=True)

# while True:
#     query=input("Enter your input: ")
#     t1=time.time()
#     l=qa_chain.invoke(query)
#     t2=time.time()
#     print("Total time: ",t2-t1)

#     def type_writing(string,delay=0.025):
#         for char in string:
#             time.sleep(delay)
#             print(char,end='',flush=True)

#     type_writing(l['result'].split("Helpful Answer:")[-1])
#     print("\n")

#     try:
#         del generate_text
#         del model
#     except:
#         pass

#     def flush():
#         gc.collect()
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()
#     flush()
