import json
from requests_futures.sessions import FuturesSession


base_url = "https://novels77.com/reaver/page-{}-{}.html"
max_pages = 56
output_file = "reaver.json"
num2 = 10034635

base_url = "https://novels77.com/heart-of-obsidian/page-{}-{}.html"
max_pages = 66
output_file = "heart_of_obsidian.json"
num2 = 85742

sess = FuturesSession()

page_urls = {
    base_url.format(page_num, num2 + page_num): page_num
    for page_num in range(1, max_pages + 1)
}

page_reqs = [sess.get(page_url) for page_url in page_urls]
page_results = [page_req.result() for page_req in page_reqs]

page_contents = {
    page_urls[page_result.url]: page_result.content.decode()
    for page_result in page_results
}

with open(output_file, "wt") as file:
    json.dump(page_contents, file, indent=2)
