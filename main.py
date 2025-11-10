import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# Load environment variables (like your DEEPSEEK_API_KEY)
load_dotenv()

# --- 1. Pydantic Models for Request and Response ---

class CodeInput(BaseModel):
    """Defines the expected JSON input from the React app."""
    code: str
    language: str

class AnalysisOutput(BaseModel):
    """Defines the JSON output structure we want from the LLM."""
    time: str
    timeExplanation: str
    space: str
    spaceExplanation: str

# --- 2. FastAPI App Initialization ---

app = FastAPI(
    title="LeetCode Complexity Analyzer API",
    description="Analyzes code complexity using DeepSeek and LangChain",
)

# Add CORS middleware to allow requests from your React frontend
# (Update origins to your frontend's URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 3. LangChain Setup ---

# Initialize the DeepSeek Chat Model
# This automatically uses the DEEPSEEK_API_KEY from your .env file
try:
    model = ChatDeepSeek(
        model="deepseek-chat", # You can specify other models if needed
        temperature=0
    )
except Exception as e:
    print(f"Error initializing DeepSeek model: {e}")
    print("Please ensure your DEEPSEEK_API_KEY is set in the .env file.")
    model = None

# Create a JSON output parser linked to our AnalysisOutput model
parser = JsonOutputParser(pydantic_object=AnalysisOutput)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert algorithm analyst. Your sole purpose is to analyze a given code snippet and return its time and space complexity in Big O notation.

You MUST provide your output *only* in the following JSON format. Do not include *any* other text, markdown, greetings, or explanations outside of the JSON structure.

{format_instructions}
"""),
    ("human", """
Analyze the following code snippet, written in {language}:

<code>
{code}
</code>
""")
])

# Format the prompt with the parser's instructions
formatted_prompt = prompt_template.partial(
    format_instructions=parser.get_format_instructions()
)

# Create the full analysis chain
# This chain will:
# 1. Take 'language' and 'code' input
# 2. Format the prompt
# 3. Send it to the DeepSeek model
# 4. Parse the model's JSON output into our AnalysisOutput format
if model:
    chain = formatted_prompt | model | parser
else:
    chain = None

# --- 4. API Endpoint ---

@app.get("/")
def read_root():
    return """
    <html>
        <head>
            <title>Welcome UpTimeRobot</title>
        </head>
        <body>
            <h1>Hi UptimeRobot ;)</h1>
        </body>
    </html>
    """

@app.post("/analyze", response_model=AnalysisOutput)
async def analyze_code(input: CodeInput):
    """
    Receives code from the frontend and returns its complexity analysis.
    """
    if not chain:
        return {
            "time": "Error",
            "timeExplanation": "Server-side model is not configured.",
            "space": "Error",
            "spaceExplanation": "Please check the server logs."
        }
        
    try:
        # Invoke the LangChain chain with the user's code
        response_json = await chain.ainvoke({
            "language": input.language,
            "code": input.code
        })
        return response_json
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {
            "time": "Error",
            "timeExplanation": "Failed to analyze code. The code may be incomplete or invalid.",
            "space": "Error",
            "spaceExplanation": f"Server error: {str(e)}"
        }

# --- 5. Run the Server ---

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
