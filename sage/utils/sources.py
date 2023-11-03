from pathlib import Path
from typing import Dict, List
from threading import Thread
from requests.exceptions import HTTPError

from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

from utils.validator import SourceData, ConfluenceModel, GitlabModel, \
    Core, Web
from constants import sources_config, core_config, EMBEDDING_MODEL
from utils.exceptions import SourceException

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
    source_dir = Path(core_config.data_dir) / "sources"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.source_dir.mkdir(exist_ok=True)
        self.source_refresh_list = list()   # type List[dict]

    @staticmethod
    def _get_source_metadata(source: SourceData | Web) -> str:
        """Generate a unique metadata for each source"""
        if isinstance(source, ConfluenceModel):
            spaces = "-".join(source.spaces)
            return f"confluence-{spaces}.source"
        elif isinstance(source, GitlabModel):
            spaces = "-".join(source.spaces)
            return f"gitlab-{spaces}.source"
        else:
            links = "-".join(source.links)
            return f"web-{links}.source"

    def _get_source_metadata_path(self, source: SourceData | Web) -> Path:
        return self.source_dir / self._get_source_metadata(source)

    def _save_source_metadata(self, source: SourceData | Web):
        """Save the source metadata to disk"""
        metadata = self._get_source_metadata_path(source)
        metadata.write_text("True")

    def _source_exist_locally(self, source: SourceData | Web):
        """Returns whether the given source exist or not"""
        return self._get_source_metadata_path(source).exists()

    def check_sources_exist(self):
        for source_name, source_data in vars(sources_config).items():
            if source_data is None:
                continue
            if not self._source_exist_locally(source_data):
                self.source_refresh_list.append({
                    "id": source_name,
                    "data": source_data
                })

    def _create_and_save_db(self, source_name: str, documents: List[Document]) -> None:
        """Creates and save a vector store index DB to file"""
        # Create vector index
        db = FAISS.from_documents(
            documents=documents,
            embedding=EMBEDDING_MODEL)

        # Save DB to source directory
        dir_path = self.source_dir / "faiss"
        db.save_local(str(dir_path), source_name)

    def _add_confluence(self, source: ConfluenceModel):
        """
        Adds and saves a confluence data source

        Args:
            source (ConfluenceModel): Confluence data model

        Raises:
            SourceException: HTTP Error communicating with confluence
        """
        try:
            loader = ConfluenceLoader(
                url=source.server,
                username=source.username,
                api_key=source.password
            )

            confluence_documents = []

            for space in source.spaces:
                documents = loader.load(
                    space_key=space,
                    include_attachments=False,
                    limit=200
                )
                confluence_documents.extend(documents)
        except HTTPError as error:
            raise SourceException(
                f"An error has occured while loading confluence source: {str(error)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len
        )

        # Split, chunk, and save the documents db
        self._create_and_save_db(
            source_name="confluence",
            documents=splitter.split_documents(confluence_documents)
        )

    def add_source(self, id: str, data: SourceData | Web) -> None:
        print(f"Processing source {id}...")
        if isinstance(data, ConfluenceModel):
            self._add_confluence(data)

        # save source metadata
        self._save_source_metadata(data)
        print("done...")

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
