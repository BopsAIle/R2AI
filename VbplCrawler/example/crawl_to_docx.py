from __future__ import annotations

import argparse
import time
from pathlib import Path

from VbplCrawler import VbplCrawler
from VbplCrawler.html_to_docx import html_to_docx, safe_filename


def _docx_exists(output_dir: Path, doc_id: str) -> bool:
    return any(output_dir.glob(f"*_{doc_id}.docx"))


def _normalize_title(title: str) -> str:
    if not title:
        return ""
    try:
        return title.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return title


def _title_contains(title: str, phrase: str) -> bool:
    normalized = _normalize_title(title)
    haystack = f"{title} {normalized}".lower()
    return phrase.lower() in haystack


def _has_valid_title(title: str) -> bool:
    normalized = _normalize_title(title).strip()
    if not normalized:
        return False
    return not all(c in "-_ " for c in normalized)


def _eff_status_matches(item: dict, eff_status: str | None) -> bool:
    if not eff_status:
        return True
    status = item.get("effStatus") or {}
    code = (status.get("code") or "").lower()
    name = _normalize_title(status.get("name") or "").lower()
    needle = eff_status.strip().lower()
    return needle in {code, name} or needle in name


def crawl_pages(
    *,
    keyword: str,
    start_page: int,
    end_page: int,
    page_size: int,
    output_dir: Path,
    request_delay: float,
    max_documents: int | None = None,
    title_contains: str | None = None,
    eff_status: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    crawler = VbplCrawler(request_delay=request_delay)

    first = crawler.search(keyword=keyword, page_number=start_page, page_size=page_size)
    print(f"Tổng số văn bản khớp: {first.total}")
    limit_text = f", tối đa {max_documents} văn bản" if max_documents else ""
    print(f"Crawl trang {start_page} đến {end_page}, mỗi trang {page_size} văn bản{limit_text}")
    if title_contains:
        print(f'Chỉ lưu văn bản có "{title_contains}" trong tên')
    if eff_status:
        print(f'Chỉ lưu văn bản có trạng thái hiệu lực: "{eff_status}"')
    print(f"Lưu file docx vào: {output_dir}")

    saved = 0
    skipped = 0
    filtered = 0
    status_filtered = 0
    no_title = 0
    collected = 0
    failed: list[tuple[str, str]] = []

    for page_number in range(start_page, end_page + 1):
        if page_number == start_page:
            result = first
        else:
            time.sleep(request_delay)
            result = crawler.search(
                keyword=keyword,
                page_number=page_number,
                page_size=page_size,
            )

        print(f"\n=== Trang {page_number}/{end_page} ({len(result.items)} văn bản) ===")

        for index, item in enumerate(result.items, start=1):
            if max_documents is not None and collected >= max_documents:
                print(f"\nĐã đủ {max_documents} văn bản, dừng crawl.")
                print()
                print(
                    f"Hoàn tất: {saved} văn bản mới, {skipped} đã có sẵn, "
                    f"{filtered} bỏ qua (tên không khớp), {status_filtered} bỏ qua (trạng thái không khớp), "
                    f"{no_title} bỏ qua (không có tên), {len(failed)} thất bại"
                )
                if failed:
                    for doc_id, message in failed[:10]:
                        print(f"  - {doc_id}: {message}")
                return

            doc_id = str(item.get("id"))
            title = _normalize_title(item.get("title") or "")
            doc_num = item.get("docNum") or ""

            if not _has_valid_title(title):
                no_title += 1
                continue

            if title_contains and not _title_contains(title, title_contains):
                filtered += 1
                continue

            if not _eff_status_matches(item, eff_status):
                status_filtered += 1
                continue

            if _docx_exists(output_dir, doc_id):
                skipped += 1
                collected += 1
                continue

            print(f"[trang {page_number} - {index}/{len(result.items)}] {doc_id} - {title}")

            try:
                time.sleep(request_delay)
                metadata = crawler.get_document(doc_id)
                html = (metadata.get("documentContent") or {}).get("content")
                if not html:
                    raise RuntimeError("Không có nội dung HTML")

                metadata_title = _normalize_title(metadata.get("title") or title)
                if not _has_valid_title(metadata_title):
                    no_title += 1
                    print("  Bỏ qua: không có tên văn bản")
                    continue
                if title_contains and not _title_contains(metadata_title, title_contains):
                    filtered += 1
                    print(f"  Bỏ qua: tên metadata không chứa \"{title_contains}\"")
                    continue
                if not _eff_status_matches(metadata, eff_status):
                    status_filtered += 1
                    print(f"  Bỏ qua: trạng thái metadata không khớp \"{eff_status}\"")
                    continue

                metadata_doc_num = metadata.get("docNum") or doc_num
                filename = safe_filename(f"{metadata_doc_num}_{metadata_title}_{doc_id}")
                docx_path = output_dir / f"{filename}.docx"
                html_to_docx(html, docx_path, title=metadata_title)
                saved += 1
                collected += 1
            except Exception as exc:
                failed.append((doc_id, str(exc)))
                print(f"  Lỗi: {exc}")

    print()
    print(
        f"Hoàn tất: {saved} văn bản mới, {skipped} đã có sẵn, "
        f"{filtered} bỏ qua (tên không khớp), {status_filtered} bỏ qua (trạng thái không khớp), "
        f"{no_title} bỏ qua (không có tên), {len(failed)} thất bại"
    )
    if failed:
        for doc_id, message in failed[:10]:
            print(f"  - {doc_id}: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl kết quả tìm kiếm vbpl.vn theo keyword và chuyển sang docx"
    )
    parser.add_argument("--keyword", default="doanh nghiệp")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--all-pages", action="store_true")
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument(
        "--title-contains",
        default="",
        help='Chỉ crawl văn bản có cụm từ này trong tên (đặt "" để tắt lọc)',
    )
    parser.add_argument(
        "--eff-status",
        default="",
        help='Chỉ crawl văn bản có trạng thái hiệu lực (vd: "CHL" hoặc "còn hiệu lực")',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục lưu file docx (mặc định: output/<keyword>-docx)",
    )
    args = parser.parse_args()
    title_contains = args.title_contains.strip() or None
    eff_status = args.eff_status.strip() or None

    crawler = VbplCrawler(request_delay=args.delay)
    probe = crawler.search(
        keyword=args.keyword,
        page_number=args.start_page,
        page_size=args.page_size,
    )
    total_pages = (probe.total + args.page_size - 1) // args.page_size

    if args.all_pages:
        end_page = total_pages
        max_documents = None
    elif args.end_page is not None:
        end_page = args.end_page
        max_documents = args.max_documents
    else:
        max_documents = args.max_documents
        end_page = args.start_page + (max_documents + args.page_size - 1) // args.page_size - 1
        end_page = min(end_page, total_pages)

    print(
        "Request body search:\n"
        f'  {{"pageNumber": <trang>, "pageSize": {args.page_size}, "keyword": "{args.keyword}"}}'
    )
    print(f"Tổng khớp keyword: {probe.total} văn bản (~{total_pages} trang)")
    if max_documents is not None:
        print(f"Giới hạn crawl: {max_documents} văn bản (trang {args.start_page}–{end_page})")
    if title_contains:
        print(f'Lọc tên văn bản: phải chứa "{title_contains}"')
    if eff_status:
        print(f'Lọc trạng thái hiệu lực: "{eff_status}"')

    output_dir = args.output_dir
    if output_dir is None:
        slug = args.keyword.strip().lower().replace(" ", "-")
        output_dir = Path(__file__).resolve().parent.parent / "output" / f"{slug}-docx"
    crawl_pages(
        keyword=args.keyword,
        start_page=args.start_page,
        end_page=end_page,
        page_size=args.page_size,
        output_dir=output_dir,
        request_delay=args.delay,
        max_documents=max_documents,
        title_contains=title_contains,
        eff_status=eff_status,
    )


if __name__ == "__main__":
    main()
