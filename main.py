import streamlit as st
import pandas as pd
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader, CSVLoader
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
import json
import re

st.title('CSV Analyzer')
apikey = ""
apikey = st.text_input("Please enter your OpenAI API Key here")
if st.button("Send", key="send_api_key"):
    if apikey.strip():
        st.success("Successfully updated api key")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

OPENAI_API_KEY = apikey
client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)

def extract_text_and_code(text):
    code_match = re.search(r'```(?:python\n)?(.*?)```', text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip() 
    else:
        code = None
    text_parts = re.split(r'```.*?```', text, flags=re.DOTALL)
    text_outside_code = "".join(text_parts).strip()

    return text_outside_code, code


def context(query:str,data)-> str:
    vector_db = Chroma.from_documents(data, embedding_model, collection_name="vector_db", persist_directory="./chromadb")
    # vector_db = Chroma(collection_name="vector_db", persist_directory="./chromadb", embedding_function=embedding_model)
    vector_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # docs = vector_db.get()
    # documents = docs["documents"]
    # keyword_retriever = BM25Retriever.from_texts(documents)
    # keyword_retriever.k = 10

    # ensemble_retriever = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.5, 0.5])

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_kwargs={"k":5}), llm=llm)

    retrieved_docs = retriever_from_llm.invoke(query)
    # print(retrieved_docs)
    context = ''
    for doc in retrieved_docs:
        context +=f"Doc content: {str(doc.page_content)}\nSource: {str(doc.metadata['source'])}"
    print(len(context))
    return context

tools = [{
            "type": "function",
            "function": {
                "name": "context_retrieve",
                "description": "This function takes a query as input and retrieves context related to the query from the dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to retrieve relavant context from database based on the query",
                        },
                    },

                    "required": ["query"],
                },
            },  
        }]

if uploaded_file is not None:
    context2 = []
    context2 = []
    context2.append({"role" : "system", "content" : '''I am a CSV file analyzer whose role is to generate analysis based on the query given by user. I always follow specific rules demilited by <>.
    Rules : 
    <>
    ~ Always retrieve the context from context_retrieve tool to get the relavant context regarding the query. Pass in the relavant query that you wish to retrieve from the context_retrieve tool.
    ~ For example if user asks you or you might encounter a situation in the analysis where you want to show a plot of y vs x. Then you query the context_retrieve tool such that you get all the information related to x and y.
    ~ After retrieving the context you perform analysis from context retrieved. 
    ~ The analysis should also include some plots demonstrating the analysis.
    ~ Note : The plot should also show the units of each axes. In the figure that the units taken into consideration should also be displayed.
    ~ To show the plot you can generate a python code that plots graphs assuming the csv file is stored as "uploaded_file.csv" at root directory. Remember that the code should be delimited within ```. Note that within ``` (triple backticks) there should only be python code and nothing else. 
    ~ The code you generate should show the plots on a streamlit application that is already running. Use st.pyplot to show the plots in the code. You should plot them such that it reflects on streamlit application.
    ~ The specific use case for your understanding is that when i seperate the code out using regular expressions and using backticks as indicators, when i run the code using "exec" function it should just show me the plots on streamlit application that is already running. So within backticks, code should only be there. Also avoid "python" at start of the code. 
    ~ Generate the code for plot in same generation. Avoid saying "Now i will plot a graph" and postponing it to next generation.
    ~ Avoid saying anything about python code. For example avoid statements like we can generate python code snippet or something like this. Just generate include python code within triple backlits and do not say anything about the code or give description about it.
     <>'''})
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_file.csv")
    loaders = [
        CSVLoader(file_path="uploaded_file.csv", encoding='utf-8', csv_args={'delimiter':","}),
        # CSVLoader(file_path="product_sales(utf-8).csv", encoding='utf-8', csv_args={'delimiter':","})
        ]

    data = []
    for loader in loaders:
        data.extend(loader.load())

    st.write(df)
    answer = st.text_input("Please enter your query here")
    
    if st.button("Send", key="send"):
        if answer.strip():
            context2.append({"role" : "user", "content" : f"Query : {answer}"})
            response = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=context2,
                    tools=tools,
                    tool_choice="auto"
                )
            response_message = response.choices[0].message
            context2.append(response_message) 
            tool_calls = response_message.tool_calls
            response_message = {"message" : response_message.content}
            if tool_calls:
                available_functions = {
                    "context_retrieve": context,
                }
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    print("Fucntion name and function argss : ",function_name, function_args)
                    if function_name == "context_retrieve":
                        # print("entered non job desc db")
                        function_response = context(query=function_args.get("query"),data = data)
                        print("function response : ",function_response)
                        response_message = function_response
                        context2.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(function_response),
                            }
                        )
                    
                        response = client.chat.completions.create(
                        model="gpt-4-0125-preview",
                        messages=context2
                        )
                        context2.append(response.choices[0].message) 
                        response_message = {"message" : response.choices[0].message.content}
                        print(response_message)
            if response_message:
                text,code = extract_text_and_code(response_message["message"])
                if text:
                    st.text(text)
                if code:
                    exec(code)
                
else:
    st.write("Please upload a CSV file to display its content.")
