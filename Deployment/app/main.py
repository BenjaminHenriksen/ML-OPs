from fastapi import FastAPI
app = FastAPI()

from http import HTTPStatus

# @app.get("/")
# def root():
#     """ Health check."""
#     response = {
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response

# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#    return {"item_id": item_id}

# from enum import Enum
# class ItemEnum(Enum):
#    alexnet = "alexnet"
#    resnet = "resnet"
#    lenet = "lenet"

# @app.get("/restric_items/{item_id}")
# def read_item(item_id: ItemEnum):
#    return {"item_id": item_id}

# @app.get("/query_items")
# def read_item(item_id: int):
#    return {"item_id": item_id}

# database = {'username': [ ], 'password': [ ]}

# @app.post("/login/")
# def login(username: str, password: str):
#    username_db = database['username']
#    password_db = database['password']
#    if username not in username_db and password not in password_db:
#       with open('database.csv', "a") as file:
#             file.write(f"{username}, {password} \n")
#       username_db.append(username)
#       password_db.append(password)
#    return "login saved"

import re

@app.get("/text_model/")
def contains_email(data: str):
   regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
   ## check domain is hotmail or gmail
   response = {
      "input": data,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
      "is_email": re.fullmatch(regex, data) is not None,
      "is_hotmail": "@hotmail." in data,
      "is_gmail": "@gmail." in data
   }

   return response

# from fastapi import UploadFile, File
# from typing import Optional
# import cv2

# @app.post("/cv_model/")
# async def cv_model(data: UploadFile = File(...), w: Optional[int] = 224, h: Optional[int] = 224):
#     with open('image.jpg', 'wb') as image:
#        content = await data.read()
#        image.write(content)
#        image.close()
    
#        img = cv2.imread('image.jpg')
#        res = cv2.resize(img, (h, w))

#     response = {
#        "input": data,
#        "message": HTTPStatus.OK.phrase,
#        "status-code": HTTPStatus.OK,
#        "w": w,
#        "h": h}
       
#     return response
