## data_loader.py
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sage.constants import SENTINEL_PATH, logger, sources_config
from sage.utils.sources import Source


async def notify_chat_service():
    """Helper for notifying the sage chat service"""
    await SENTINEL_PATH.write_text("updated")


async def update_sources():
    """A task for updating the sources on schedule"""
    logger.info("Triggering sources update")
    await Source().run()
    # Send an update signal to the chat service
    await notify_chat_service()


async def main():
    """Main Executor"""
    scheduler_time = sources_config.refresh_schedule
    if scheduler_time is None:
        logger.warn("Data Loader Scheduler won't run. No schedule interval configured.")
        return
    interval = CronTrigger.from_crontab(scheduler_time)
    scheduler = AsyncIOScheduler(gconfig={"logger": logger})
    scheduler.add_job(update_sources, trigger=interval)
    scheduler.start()
    logger.info("Data Loader Scheduler started, now running forever.")
    while True:
        await asyncio.sleep(1000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
