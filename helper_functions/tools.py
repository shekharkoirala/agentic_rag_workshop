from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.utilities import WikipediaAPIWrapper
import io
from PIL import Image
import requests
from helper_functions.keys import WEATHER_KEY, HUGGING_FACE_KEY

WIKI_API_WRAPPER = WikipediaAPIWrapper(top_k_results=1)

# ------------------------------------------------------------------------------------
##### Wikipedia tool #####
def wikipedia_caller(query:str) ->str:
    """This function queries wikipedia through a search query."""
    return WIKI_API_WRAPPER.run(query)

# Input parameter definition
class QueryInput(BaseModel):
    query: str = Field(description="Input search query")

# the tool description
wiki_tool_description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

# fuse the function, input parameters and description into a tool. 
my_own_wiki_tool = StructuredTool.from_function(
    func=wikipedia_caller,
    name="wikipedia",
    description=wiki_tool_description,
    args_schema=QueryInput,
    return_direct=False,
)

# ------------------------------------------------------------------------------------
##### Weather tool #####
def extract_city_weather(city:str)->str:
    api_key = WEATHER_KEY

    # Build the API URL
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?key={api_key}&unitGroup=metric"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        current_temp = data['days'][0]['temp']
        output = f"Current temperature in {city}: {current_temp}°C"
    else:
        output = f"Error: {response.status_code}"

    return output

# Input parameter definition
class WeatherInput(BaseModel):
    city: str = Field(description="City name")


# the tool description
weather_tool_description: str = (
        """
        Allows to extract the current temperature in a specific city. 
        """
    )

# fuse the function, input parameters and description into a tool. 
weather_tool = StructuredTool.from_function(
    func=extract_city_weather,
    name="weather",
    description=weather_tool_description,
    args_schema=WeatherInput,
    return_direct=False,
)
# ------------------------------------------------------------------------------------
##### Image tool #####
def text_to_image(payload:str):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_KEY}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    
    image_bytes = query({
        "inputs": payload,
    })

    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image
    new_size = (400, 400)  # Example new size (width, height)
    resized_image = image.resize(new_size)



    # Save the resized image to a file
    image_path = f'images/image_{payload.replace(" ", "_")}.jpg'
    resized_image.save(image_path)
    
    # Return the path to the saved image
    return f'{image_path} '


# Input parameter definition
class ImageInput(BaseModel):
    payload: str = Field(description="What should be converted into image")


# the tool description
images_tool_description: str = (
       "Generate an image based on the input text and return its path"
    )

# fuse the function, input parameters and description into a tool. 
image_tool = StructuredTool.from_function(
    func=text_to_image,
    name="create_image",
    description=images_tool_description,
    args_schema=ImageInput,
    return_direct=False,
)

# ------------------------------------------------------------------------------------
##### TODO: ADD YOUR OWN TOOLS HERE #####