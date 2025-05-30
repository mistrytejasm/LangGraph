from json import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults # inbuild tool for search
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(
      model="gemini-2.0-flash",
      # temperature = 0.7,
      # max_tokens=None,
      # timeout=None,
      # max_retries=2
      )

# underthe hood do the google search 
search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
  """Return the current date and time in the specified format"""
  current_time = datetime.datetime.now()
  formatted_time =current_time.strftime(format)
  return formatted_time

tools = [search_tool, get_system_time]

# create a agent
agent = initialize_agent(tools = tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("when was the spaceX's last launch and how many days ago was the that from this instance")


# result = llm.invoke("give me a tweet about today's weather in mumbai")
# print(result.content)
