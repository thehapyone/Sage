from utils.validator import SourceData, Core
from constants import sources_config


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

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        pass
        
    def run(self):
        return None