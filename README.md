# API SUMMARIZER

It`s a simple web application that integrates an AI-powered text summarization service using FastAPI and LangChain. The application should accept text input from the user and return a summarized version of the text.

## Instruction

1. Setup the Environment:
  - Create a virtual environment for your project.
  - Install requirements by command <code>pip install -r requirements.txt</code>

2. Run application 
 - <code>uvicorn main:app --reload</code>

<hr>

### Endpoint

`POST /summarize`

### Request

- **URL**: `http://127.0.0.1:8000/summarize`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Body**: JSON object with a `text` field containing the text to be summarized.

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '{"text": "Your text to be summarized goes here."}'
```

#### Response
- Success (200 OK): Returns a JSON object with the summarized text. 
- Error (400 Bad Request): Returns a JSON object with an error message.