from VbplCrawler import VbplCrawler


def main() -> None:
    crawler = VbplCrawler(request_delay=0.2)

    # 1) Search
    result = crawler.search(
        keyword="doanh nghiệp",
        page_number=1,
        page_size=100,
    )
    print(f"Tổng số văn bản khớp: {result.total}")
    print(f"Trang 1 có {len(result.items)} văn bản")

    if not result.items:
        return

    first_id = result.items[0]["id"]
    print(f"Văn bản đầu tiên: {first_id} - {result.items[0]['title']}")

    # 2) Lấy nội dung HTML đầy đủ
    html = crawler.get_document_html(first_id)
    print(f"Độ dài HTML: {len(html)} ký tự")

    # 3) Parse các yếu tố prov-chapter / prov-article / prov-clause / ...
    elements = crawler.parse_provision_elements(html)
    print(f"Số phần tử cấu trúc: {len(elements)}")
    for element in elements[:5]:
        print(f"- [{element['type']}] {element['text'][:120]}")

    # 4) Crawl nhỏ để test (bỏ max_documents để crawl toàn bộ)
    docs = crawler.crawl_keyword(
        keyword="doanh nghiệp",
        page_size=100,
        max_pages=1,
        max_documents=3,
        output_dir="output",
    )
    print(f"Đã lưu {len(docs)} văn bản vào thư mục output/")


if __name__ == "__main__":
    main()
