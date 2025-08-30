import httpx
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

APP_URL=""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ping():
    try:
        async with httpx.AsyncClient() as client:
            response =await client.get(APP_URL)
            response.raise_for_status()
            logger.info(f"Ping successful: {response.status_code}")
    except httpx.HTTPError as e:
        logger.error(f"Ping fail: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler=AsyncIOScheduler()
    trigger=IntervalTrigger(minutes=30)
    scheduler.add_job(ping,trigger=trigger)
    scheduler.start()
    logging.info("Scheduler started")
    yield
    scheduler.shutdown()
    logging.info("Scheduler shutting down")

app = FastAPI(lifespan=lifespan)

@app.get('/')
async def root():
    return {'message':"Server running"}