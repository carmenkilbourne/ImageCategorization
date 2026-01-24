from fastapi import FastAPI

app = FastAPI(title="Hello world")

@app.get("/")
def index():
    return {"message": "This is IMDb Reviews Classification API!"}
