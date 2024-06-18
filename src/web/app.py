import os
import dotenv
import pandas as pd
from io import StringIO
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import streamlit as st
from langchain import agents
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import random
import json
import pytz
from datetime import datetime

dotenv.load_dotenv()

st.title("💬 AI agent that can use a DSL to calculate")
st.caption("🚀 A Bot that can use a DSL to execute logic")

def get_session_id() -> str:
    id = random.randint(0, 1000000)
    return "00000000-0000-0000-0000-" + str(id).zfill(12)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = get_session_id()
    print("started new session: " + st.session_state["session_id"])
    st.write("You are running in session: " + st.session_state["session_id"])

llm: AzureChatOpenAI = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        streaming=True
    )
else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True
    )

@tool
def time_for_loading(input: str) -> int:
    "Calculate the time for loading the given weight. The weight is an integer in kilograms. The input should be only a number"
    input = str.replace(input, "\n", "")
    input = str.replace(input, "\"", "")
    input = str.replace(input, " ", "")
    print("calculate time for loading for weight: ", input)
    # jsoninput = json.loads(input)
    # weight = jsoninput['weight']
    weight = int(input)
    if weight <= 2:
        return 1
    elif weight <= 6:
        return 3
    elif weight <= 10:
        return 4
    else:
        return 10
    
@tool
def calculate_travel_time(input: str) -> int:
    "Calculate the time for traveling the given weight and distance. The weight is in kilograms and the distance is in kilometers. The input should just be a set of numbers of the format '23, 23'"
    input = str.replace(input, "\n", "")
    input = str.replace(input, "\"", "")
    input = str.replace(input, " ", "")
    print ("calculate time for traveling for weight and distance: ", input)
    weightstr, distancestr = input.split(",")
    weight = int(weightstr)
    distance = int(distancestr)
    print("calculate time for traveling for weight: ", weight, " and distance: ", distance)
    if weight <= 2:
        return 1.5 * distance
    elif weight <= 6:
        return 3 * distance
    elif weight <= 10:
        return 4 * distance
    else:
        return 10 * distance

@tool
def get_current_time(location: str) -> str:
    "Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin"
    try:
        print("get current time for location: ", location)
        location = str.replace(location, " ", "")
        location = str.replace(location, "\"", "")
        location = str.replace(location, "\n", "")
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the timezone for that location."
    

tools = [time_for_loading, calculate_travel_time, get_current_time]

commandprompt = '''
    ##
    You are a logistic agent for calculating time for shipping cargo of boxes using a truck. 
    You need to perform the following tasks based on the User query. 
    The task aims to create commands and provide the commands as output.
    Only one box can be loaded or unloaded at a time.
    You should calculate the weight of the truck before every time a box has to be loaded to make sure you do no exceed the maximum weight.
    It is most important to not exceed the maximum weight of the truck or the maximum number of boxes that can be loaded onto the truck. Distribute the boxed accordingly and load the heaviest boxes first.

    If you are not able to understand the User query. Take a deep breath, think step by step. 
    Despite deliberation, if you are not able to create commands. Just answer with not able to create commands.
    The grammar defines several commands for shipping cargo. Each command takes specific arguments. 
    The '%Y-%m-%d %H:%M:%S' means string formatted datetime format.

    A blue box weighs 5 kilograms and has dimensions. A red box weighs 10 kilograms. A green box weighs 15 kilograms.
    A truck can carry a maximum of 5 boxes of cargo or load a maximum of 50 kilograms.
    It takes 3 minutes to load or unload a box onto a truck. Only a single box can be loaded or unloaded at a time.
    
    Use the tool calculate_travel_time to calculate the traveling time of a truck. The function takes the weight of the cargo in kilograms and the distance in kilometers as arguments.

    You are able to create the follwing commands:

    The `prepare_truck` command takes new truck and gives it an unique identifier.
        `prepare_truck (truck_id)`
    
    The `load_box_on_truck` command takes a weight of the box as argument and adds a box to the truck. The weight is a number that represents the weight of the cargo in kilograms.
        `load_box_on_truck (truck_id, box_id, weight)`

    The `calculate_weight_of_truck` command calculated the weight of all the boxes in the truck. The weight is a number that represents the weight of the cargo in kilograms.
        `calculate_weight_of_truck (truck_id)`

    The `drive_truck_to_location` command takes a weight of the cargo in kilograms and the distance in kilometers. The weight is a number that represents the weight of the cargo in kilograms.
        `drive_truck_to_location (truck_id, weight, distance)`
    
    The `unload_box_from_truck` command takes a weight of the box as argument and unloads a box from the truck. The weight is a number that represents the weight of the cargo in kilograms.
        `unload_box_from_truck (truck_id, box_id, weight)`

    ## Here are some examples of user inputs that you can use to generate the commands defined by the grammar:

    1. For preparing a truck:
    "Please prepare a truck with ID 42."

    2. For loading a box on a truck:
    "Please load a blue box with ID 123 on the truck with ID 42."
    "Please load a red box with ID 43 on the truck with ID 42."
    "Please load a red box with ID 44 on the truck with ID 42."

    3. For calculating the weight of the truck:
    "Please calculate the weight of the truck with ID 42 after loading the boxes."
    
    4. For driving a truck to a location:
    "Please drive truck with ID 42 to the location 100 kilometers away."

    5. For unloading a box from a truck:
    "Please unload the blue box with ID 123 from the truck with ID 42."

    Remember to replace the weights, distance, dates, times, and IDs with your actual data. Also update the weight of the truck after every time a box has been loaded or unloaded.
    The dates and times should be in the format '%Y-%m-%d %H:%M:%S'.

    ## Here are some examples of how the output might look like based on the functions you provided:

    1. For preparing a truck:
    `prepare_truck("42")`

    2. For loading a box on a truck:
    `load_box_on_truck("42", "123", 5)`
    `load_box_on_truck("42", "43", 10)`
    `load_box_on_truck("42", "44", 10)`

    3. For calculating the weight of the truck:
    `calculate_weight_of_truck("42")`

    4. For driving a truck to a location:
    `drive_truck_to_location("42", 25, 100)`

    5. For unloading a box from a truck:
    `unload_box_from_truck("42", "123", 5)`

    ## Your response ought to be the command only as follows examples. However, you can prompt for input to provide the command parameters.

    1. `prepare_truck("42")`
    2. `load_box_on_truck("42", "123", 5)`
    3. `calculate_weight_of_truck("42")`
    4. `drive_truck_to_location("42", 25, 100)`
    5. `unload_box_from_truck("42", "123", 5)`

    ##

    Make sure that the input for the time_for_loading and calculate_travel_time functions are integers for weight and distance. Make sure that you use the correct types for the input arguments which should be just a number.
    The get_current_time function takes a string as input for the location. The location should be in a format like America/New_York, Asia/Bangkok, Europe/London.
    The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London
    
    '''

promptString = commandprompt +  """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]. Make sure that Actions are not commands. They should be the name of the tool to use.

Action Input: the input to the action according to the tool signature

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}

"""
prompt = PromptTemplate.from_template(promptString)

agent = create_react_agent(llm, tools, prompt)
agent_executor = agents.AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


if prompt := st.chat_input():

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
