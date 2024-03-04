## data_loader.py
## Helper module that helps to load datasource changes to the configuration file

import asyncio
from sage.utils.sources import Source

asyncio.run(Source().run())
