from pathlib import Path
from typing import Dict, List
from threading import Thread

from langchain.document_loaders import ConfluenceLoader

from utils.validator import SourceData, Core, Web
from constants import sources_config, core_config


#  - call load_sources
#  	- from load_source
#  		 - loop through all sources
#  		 	- check if source is already saved
#  		 	- if yes, exit the source thread
#  		 	- if no,
#  		 		- for confluence source
#  		 			- find all pages add them to a confluence page queue list
#  		 		- for gitlab source
#  		 			- find all projects in the group and add them to gitlab_source list
#  		 		- for web source
#  		 			- find all links in the web.link and add them to a links queue/list

class Source:

    _instance = None
    source_dir = Path(core_config.sources_dir)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.source_dir.mkdir(exist_ok=True)
        self.source_refresh_list = list()   # type List[dict]

    def _source_exist_locally(self, source_name: str):
        local_source = self.source_dir / source_name / ".txt"
        return local_source.exists()

    def check_sources_exist(self):
        for source_name, source_data in vars(sources_config).items():
            if source_data is None:
                continue
            if not self._source_exist_locally(source_name):
                self.source_refresh_list.append({
                    "id": source_name,
                    "data": source_data
                })

    @staticmethod
    def _add_confluence(source: SourceData):
        # Add confluence datasource
        space_id = source.spaces[0]
        loader = ConfluenceLoader(
            url=source.server,
            username=source.username,
            api_key=source.password
        )
        documents = loader.load(
            space_key=space_id, include_attachments=False, limit=100)


    def add_source(self, id: str, data: SourceData | Web):
        print(f"Processing source {id}...")
        if isinstance(data, SourceData):
            if id == "confluence":
                self._add_confluence(data)

        # save source metadata
        print("done...")
        return None

    def run(self):
        self.check_sources_exist()

        if len(self.source_refresh_list) == 0:
            print("No changes to sources")

        threads = []
        for source in self.source_refresh_list:
            thread = Thread(target=self.add_source, kwargs=source)
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
