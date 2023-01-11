from fastapi import FastAPI
app = FastAPI()

from http import HTTPStatus

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/items/{item_id}")
def read_item(item_id: int):
   return {"item_id": item_id}