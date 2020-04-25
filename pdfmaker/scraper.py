import json
from requests_futures.sessions import FuturesSession

sess = FuturesSession()


base_url = "https://readonlinefreenovel.com/patricia-briggs/silence-fallen"
page_urls = {
    "_".join([base_url, str(page_num)]): page_num
    for page_num in range(2, 100)
}
page_urls[base_url] = 1


page_reqs = [sess.get(page_url) for page_url in page_urls]
page_results = [page_req.result() for page_req in page_reqs]

page_contents = {
    page_urls[page_result.url]: page_result.content.decode()
    for page_result in page_results
}

with open("book.json", "wt") as file:
    json.dump(page_contents, file, indent=2)