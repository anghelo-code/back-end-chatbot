import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.models import router as models_router


origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    "http://localhost:9099",
    "http://localhost:53288"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(models_router, prefix='/chatbot')

@app.get("/")
def read_root():
  return {"Hello": "World"}


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        reload=True
    )
