const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const { z } = require("zod");
const { ChatDeepSeek } = require("@langchain/deepseek");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const { StructuredOutputParser } = require("@langchain/core/output_parsers");

// Load environment variables
dotenv.config();

// --- 1. Zod Models (Equivalent to Pydantic) ---

// Input Validation Schema
const CodeInputSchema = z.object({
  code: z.string(),
  language: z.string(),
});

// Output Validation Schema (Defines the JSON structure we want from the LLM)
const AnalysisOutputSchema = z.object({
  time: z.string().describe("The time complexity (e.g., O(n))"),
  timeExplanation: z.string().describe("Explanation of the time complexity"),
  space: z.string().describe("The space complexity (e.g., O(1))"),
  spaceExplanation: z.string().describe("Explanation of the space complexity"),
});

// --- 2. Express App Initialization ---

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(express.json()); // Parses incoming JSON requests (like FastAPI's auto-parsing)
app.use(
  cors({
    origin: "*", // Allows all origins
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  }),
);

// --- 3. LangChain Setup ---

let chain = null;

try {
  // Initialize the DeepSeek Chat Model
  // Ensure DEEPSEEK_API_KEY is in your .env file
  const model = new ChatDeepSeek({
    model: "deepseek-chat",
    temperature: 0,
    apiKey: process.env.DEEPSEEK_API_KEY,
  });

  // Create a parser linked to our Zod schema
  // This replaces JsonOutputParser(pydantic_object=...)
  const parser = StructuredOutputParser.fromZodSchema(AnalysisOutputSchema);

  // Define the prompt template
  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      `
You are an expert algorithm analyst. Your sole purpose is to analyze a given code snippet and return its time and space complexity in Big O notation.

You MUST provide your output *only* in the following JSON format. Do not include *any* other text, markdown, greetings, or explanations outside of the JSON structure.

{format_instructions}
    `,
    ],
    [
      "human",
      `
Analyze the following code snippet, written in {language}:

<code>
{code}
</code>
    `,
    ],
  ]);

  // Create the chain
  // In JS LangChain, we use .pipe() instead of the | operator
  chain = promptTemplate.pipe(model).pipe(parser);
} catch (e) {
  console.error("Error initializing DeepSeek model:", e);
  console.log("Please ensure your DEEPSEEK_API_KEY is set in the .env file.");
}

// --- 4. API Endpoints ---

// Root Endpoint
app.get("/", (req, res) => {
  res.send(`
    <html>
        <head>
            <title>Welcome UpTimeRobot</title>
        </head>
        <body>
            <h1>Hi UptimeRobot ;)</h1>
        </body>
    </html>
  `);
});

// Analyze Endpoint
app.post("/analyze", async (req, res) => {
  // Check if chain is initialized
  if (!chain) {
    return res.status(503).json({
      time: "Error",
      timeExplanation: "Server-side model is not configured.",
      space: "Error",
      spaceExplanation: "Please check the server logs.",
    });
  }

  try {
    // 1. Validate Input (Pydantic equivalent)
    const { code, language } = CodeInputSchema.parse(req.body);

    // 2. Invoke LangChain
    const response = await chain.invoke({
      language: language,
      code: code,
      format_instructions:
        await StructuredOutputParser.fromZodSchema(
          AnalysisOutputSchema,
        ).getFormatInstructions(),
    });

    // 3. Return JSON response
    res.json(response);
  } catch (error) {
    // Handle Validation Errors (Zod)
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Validation Error",
        details: error.errors,
      });
    }

    // Handle LLM or Server Errors
    console.error("Error during analysis:", error);
    res.status(500).json({
      time: "Error",
      timeExplanation:
        "Failed to analyze code. The code may be incomplete or invalid.",
      space: "Error",
      spaceExplanation: `Server error: ${error.message}`,
    });
  }
});

// --- 5. Run the Server ---

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
