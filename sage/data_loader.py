## data_loader.py
import asyncio
import aiofiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sage.utils.sources import Source
import asyncio

from sage.utils.sources import Source
import asyncio
from sage.utils.sources import Source
from sage.constants import logger, sources_config

async def notify_chat_service():
    """Helper for notifying the sage chat service"""
    sentinel_file_path = "/home/ayo/Desktop/codes/codesage/sandbox/data_updated.flag"
    async with aiofiles.open(sentinel_file_path, "w") as sentinel_file:
        await sentinel_file.write("updated")


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

# import os
# from sage.utils.source_qa import SourceQAService
# import chainlit as cl
# import asyncio

# # ... existing chat service code ...

# async def check_for_data_updates():
#     sentinel_file_path = '/home/appuser/data/data_updated.flag'
#     while True:
#         if os.path.exists(sentinel_file_path):
#             # Read the sentinel file
#             with open(sentinel_file_path, 'r') as sentinel_file:
#                 content = sentinel_file.read()
#             if content == 'updated':
#                 # Reload the data
#                 print("Data update detected, reloading...")
#                 # Implement the logic to reload the data in the chat service

#                 # Reset the sentinel file
#                 with open(sentinel_file_path, 'w') as sentinel_file:
#                     sentinel_file.write('')

#         # Sleep for some time before checking again
#         await asyncio.sleep(30)  # Check every 30 seconds for example

# # Start the check for data updates in parallel with the chat service
# asyncio.create_task(check_for_data_updates())

# # ... existing code to start the chat service ...
