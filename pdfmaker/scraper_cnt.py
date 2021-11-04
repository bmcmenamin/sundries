import json
from requests_futures.sessions import FuturesSession


base_url = "https://continuous.epub.pub/epub/5da692b1d7d37700070f65f5"
output_file = "reaper.json"


sess = FuturesSession()

page_urls = dict()
page_urls[base_url] = 1

page_reqs = [sess.get(page_url) for page_url in page_urls]
page_results = [page_req.result() for page_req in page_reqs]

page_contents = {
    page_urls[page_result.url]: page_result.content.decode()
    for page_result in page_results
}

with open(output_file, "wt") as file:
    json.dump(page_contents, file, indent=2)
