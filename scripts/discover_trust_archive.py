from __future__ import annotations

import concurrent.futures
import html
import re
import urllib.request


def fetch(item_id: int) -> tuple[int, str]:
    url = (
        "https://web.archive.org/web/20191210164551id_/"
        f"https://www.trust-i.net/downloads/index.php?iddesc={item_id}"
    )
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return item_id, f"ERROR: {exc}"
    match = re.search(
        r'<div id="cabecera_desc".*?<span[^>]*><a[^>]*>(.*?)</a>',
        body,
        flags=re.DOTALL,
    )
    title = re.sub(r"<[^>]+>", "", match.group(1)) if match else "title not found"
    return item_id, html.unescape(" ".join(title.split()))


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch, range(1, 149))
    for item_id, title in results:
        terms = ("watermet", "dss", "uws performance", "software", "metabolism")
        if any(term in title.lower() for term in terms):
            print(f"{item_id}: {title}")


if __name__ == "__main__":
    main()
