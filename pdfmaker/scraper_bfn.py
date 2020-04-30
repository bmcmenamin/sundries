import json
from requests_futures.sessions import FuturesSession


base_url = "https://series.bookfrom.net/nalini-singh/{}-ocean_light.html"
max_pages = 37
output_file = "ocean_light.json"
book_id = 32226


base_url = "https://readfrom.net/larissa-ion/{}-reaper.html"
max_pages = 31
output_file = "reaper.json"
book_id = 513284



sess = FuturesSession()

page_urls = {
    base_url.format("page,{},{}".format(page_num, book_id)): page_num
    for page_num in range(2, max_pages + 1)
}
page_urls[base_url.format(book_id)] = 1

page_reqs = [sess.get(page_url) for page_url in page_urls]
page_results = [page_req.result() for page_req in page_reqs]


page_contents = {}
for page_result in page_results:
    n = page_urls[page_result.url]
    page_contents[n] = page_result.content.decode('utf8', 'ignore')

with open(output_file, "wt") as file:
    json.dump(page_contents, file, indent=2)
