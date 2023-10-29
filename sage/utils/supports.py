import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
from time import sleep
import logging
from queue import Queue
from threading import Lock, Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_links(base_url: str, file_name: str):
    """
    Helps to find child links from a given link source

    Args:
        url (str): A valid URL
    """
    def normalize_url(url):
        """Helps to normalize urls by removing ending slash"""
        if url.endswith('/'):
            return url[:-1]
        return url

    def worker():
        """Worker function to process URLs from the queue"""
        while True:
            with lock:
                if not to_visit_links.empty():
                    url, depth = to_visit_links.get()
                else:
                    break
            extract_links(url, depth)

    def extract_links(url, depth):
        """Extracts all unique links from a webpage and adds them to the queue"""
        url = normalize_url(urldefrag(url)[0])

        with lock:
            if url in visited_links or base_url not in url:
                return
            visited_links.add(url)

        if depth > 10:
            logger.warning(f"Max depth reached - {url}")
            return

        logger.info(f"Scraping {url}")

        with open(file_name, 'a') as f:
            f.write(url + '\n')

        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.RequestException, requests.exceptions.SSLError) as e:
            logger.error(f"Error occurred: {e}, url: {url}")
            return
        except Exception as e:
            logger.error(f"An unexpected error has occurred: {e}, url: {url}")
            return

        soup = BeautifulSoup(response.text, 'lxml')

        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            if link.startswith(('mailto:', 'javascript:', '#')) \
                    or link.endswith(('.png', '.svg', '.jpg', '.jpeg', '.gif')):
                continue
            absolute_link = urljoin(url, link)

            with lock:
                if absolute_link not in visited_links:
                    to_visit_links.put((absolute_link, depth + 1))

        sleep(0.5)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100, pool_maxsize=100, max_retries=3)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/89.0.4389.82 Safari/537.36'
    }

    visited_links = set()
    to_visit_links = Queue()
    lock = Lock()
    base_url = normalize_url(base_url)
    to_visit_links.put((base_url, 0))

    threads = []

    for _ in range(5):
        t = Thread(target=worker)
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        with lock:
            if to_visit_links.qsize() >= 2 and len(threads) < 10:
                t = Thread(target=worker)
                t.start()
                threads.append(t)
        sleep(0.5)

    for t in threads:
        t.join()

    return visited_links
