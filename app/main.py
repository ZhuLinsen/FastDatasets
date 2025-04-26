from fastapi import FastAPI
from app.core.logger import logger
from app.api.dataset import router as dataset_router
from app.core.config import config

app = FastAPI(title="FastDatasets")
 
# 直接使用/api前缀，不再需要v1
app.include_router(dataset_router, prefix="/api")


@app.get("/")
async def root():
    logger.info("API服务已启动")
    return {"message": "FastDatasets API服务运行中"}

if __name__ == "__main__":
    import uvicorn
    logger.info("正在启动API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)