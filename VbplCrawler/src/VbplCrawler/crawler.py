from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import requests
from bs4 import BeautifulSoup, Tag

SEARCH_ACTION_ID = "c529d164f28418e5898a834422629e64c6816af1"
GATEWAY_BASE_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api"
SITE_URL = "https://vbpl.vn/"

PROVISION_CLASSES = (
    "prov-chapter",
    "prov-article",
    "prov-clause",
    "prov-item",
    "prov-content",
)


@dataclass
class SearchResult:
    total: int
    page_number: int
    page_size: int
    tokens: list[str]
    items: list[dict[str, Any]]


class VbplCrawler:
    def __init__(
        self,
        *,
        request_delay: float = 0.3,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )

    def search(
        self,
        *,
        keyword: str,
        page_number: int = 1,
        page_size: int = 100,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> SearchResult:
        payload = {
            "pageNumber": page_number,
            "pageSize": page_size,
            "keyword": keyword,
        }
        if extra_params:
            payload.update(extra_params)

        response = self.session.post(
            SITE_URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/x-component",
                "Next-Action": SEARCH_ACTION_ID,
                "Origin": "https://vbpl.vn",
                "Referer": SITE_URL,
            },
            data=json.dumps([payload]),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = self._parse_rsc_payload(response.text)
        return SearchResult(
            total=int(data.get("total", 0)),
            page_number=int(data.get("pageNumber", page_number)),
            page_size=int(data.get("pageSize", page_size)),
            tokens=list(data.get("tokens") or []),
            items=list(data.get("items") or []),
        )

    def iter_search_pages(
        self,
        *,
        keyword: str,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> Iterator[SearchResult]:
        first = self.search(
            keyword=keyword,
            page_number=1,
            page_size=page_size,
            extra_params=extra_params,
        )
        yield first

        if first.total == 0:
            return

        total_pages = (first.total + page_size - 1) // page_size
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)

        for page_number in range(2, total_pages + 1):
            time.sleep(self.request_delay)
            yield self.search(
                keyword=keyword,
                page_number=page_number,
                page_size=page_size,
                extra_params=extra_params,
            )

    def get_document(self, document_id: str | int) -> dict[str, Any]:
        response = self.session.get(
            f"{GATEWAY_BASE_URL}/qtdc/public/doc/{document_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise RuntimeError(
                f"Failed to fetch document {document_id}: {body.get('message')}"
            )
        return body["data"]

    def get_document_html(self, document_id: str | int) -> str:
        document = self.get_document(document_id)
        content = document.get("documentContent") or {}
        html = content.get("content")
        if not html:
            raise RuntimeError(f"Document {document_id} has no HTML content")
        return html

    def parse_provision_elements(
        self,
        html: str,
        *,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        elements: list[dict[str, Any]] = []

        for node in soup.find_all(["p", "div", "span", "h1", "h2", "h3", "h4"]):
            if not isinstance(node, Tag):
                continue

            classes = set(node.get("class") or [])
            matched = sorted(classes.intersection(PROVISION_CLASSES))
            if not matched:
                continue

            style = (node.get("style") or "").replace(" ", "").lower()
            if not include_hidden and "display:none" in style:
                continue

            text = node.get_text(" ", strip=True)
            if not text:
                continue

            elements.append(
                {
                    "id": node.get("id"),
                    "classes": matched,
                    "type": matched[0],
                    "parent_id": node.get("parent-id"),
                    "text": text,
                    "html": str(node),
                }
            )

        return elements

    def crawl_keyword(
        self,
        *,
        keyword: str,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        max_documents: Optional[int] = None,
        fetch_content: bool = True,
        parse_elements: bool = True,
        include_hidden: bool = False,
        output_dir: Optional[str | Path] = None,
    ) -> list[dict[str, Any]]:
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        collected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for page in self.iter_search_pages(
            keyword=keyword,
            page_size=page_size,
            max_pages=max_pages,
        ):
            for item in page.items:
                doc_id = str(item.get("id"))
                if not doc_id or doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                record: dict[str, Any] = {
                    "id": doc_id,
                    "search_item": item,
                }

                if fetch_content:
                    time.sleep(self.request_delay)
                    metadata = self.get_document(doc_id)
                    record["metadata"] = self._strip_content_fields(metadata)

                    html = (metadata.get("documentContent") or {}).get("content")
                    if html:
                        record["html"] = html
                        if parse_elements:
                            record["elements"] = self.parse_provision_elements(
                                html,
                                include_hidden=include_hidden,
                            )

                collected.append(record)

                if output_path:
                    doc_path = output_path / f"{doc_id}.json"
                    doc_path.write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                if max_documents is not None and len(collected) >= max_documents:
                    return collected

        if output_path:
            summary = {
                "keyword": keyword,
                "page_size": page_size,
                "document_count": len(collected),
                "document_ids": [doc["id"] for doc in collected],
            }
            (output_path / "_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return collected

    @staticmethod
    def _parse_rsc_payload(response_text: str) -> dict[str, Any]:
        chunks: dict[str, str] = {}
        current_index: Optional[str] = None

        for line in response_text.splitlines():
            match = re.match(r"^(\d+):(.*)", line)
            if match:
                current_index = match.group(1)
                chunks[current_index] = match.group(2)
                continue
            if current_index is not None:
                chunks[current_index] += line

        payload = chunks.get("1")
        if not payload:
            raise ValueError("Could not parse vbpl search response")

        return json.loads(payload)

    @staticmethod
    def _strip_content_fields(metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(metadata)
        content = cleaned.get("documentContent")
        if isinstance(content, dict) and "content" in content:
            cleaned["documentContent"] = {
                key: value for key, value in content.items() if key != "content"
            }
            cleaned["documentContent"]["content_length"] = len(content["content"])
        return cleaned
