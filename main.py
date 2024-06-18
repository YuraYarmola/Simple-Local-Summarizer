from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import JSONResponse
from summarizer import Summarizer

app = FastAPI()


# method to make summarization of text
@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if text:
        summary = Summarizer().summarize(text)
        return JSONResponse({"summary": summary}, 200)
    else:
        return JSONResponse({"error": "No text"}, 400)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
