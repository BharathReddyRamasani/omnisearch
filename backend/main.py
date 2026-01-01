# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from backend.app import app as api_app

# # app = FastAPI(
# #     title="OmniSearch AI 🚀",
# #     description="EDA + ETL + AutoML Platform",
# #     version="1.0.0"
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # app.mount("/api", api_app)

# # @app.get("/")
# # def root():
# #     return {
# #         "message": "🚀 OmniSearch AI Backend Running",
# #         "docs": "/api/docs",
# #         "status": "healthy"
# #     }
# # backend/main.py

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from backend.app import app as api_app

# app = FastAPI(
#     title="OmniSearch AI 🚀",
#     description="Enterprise EDA + ETL + AutoML Platform",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.mount("/api", api_app)

# @app.get("/")
# def root():
#     return {
#         "status": "healthy",
#         "message": "OmniSearch AI backend running",
#         "docs": "/api/docs"
#     }

from fastapi import FastAPI
from backend.app import app as api_app

app = FastAPI(title="OmniSearch AI Gateway")

app.mount("/api", api_app)

@app.get("/")
def root():
    return {
        "service": "OmniSearch AI",
        "status": "running",
        "docs": "/api/docs"
    }
