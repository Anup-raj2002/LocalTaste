from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import CORS_ORIGINS
from routers import recommend, sentiment

app = FastAPI(title="LocalTaste API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api")
app.include_router(sentiment.router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "LocalTaste API running"}