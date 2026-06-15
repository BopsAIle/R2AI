# VbplCrawler

Crawler dữ liệu từ [vbpl.vn](https://vbpl.vn/).

## Cách hoạt động

1. **Tìm kiếm** qua Next.js Server Action của vbpl.vn với body:

```json
{
  "pageNumber": 1,
  "pageSize": 100,
  "keyword": "doanh nghiệp"
}
```

2. **Lấy nội dung văn bản** qua API gateway:

```
GET https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{id}
```

Trường `documentContent.content` chứa HTML đầy đủ với các class như `prov-chapter`, `prov-article`, `prov-clause`, `prov-item`, `prov-content`.

## Cài đặt

```bash
pip install -e .
```

## Ví dụ

```bash
python example/example.py
```

## Crawl toàn bộ kết quả tìm kiếm

```python
from VbplCrawler import VbplCrawler

crawler = VbplCrawler(request_delay=0.3)

docs = crawler.crawl_keyword(
    keyword="doanh nghiệp",
    page_size=100,
    output_dir="output/doanh-nghiep",
)
```

Với từ khóa `doanh nghiệp`, hiện có khoảng **53.614** văn bản (537 trang, mỗi trang 100 bản ghi). Nên đặt `request_delay` hợp lý để tránh gây tải cho server.

## Lưu ý

- API search không phải REST công khai trực tiếp; crawler dùng đúng server action mà frontend vbpl.vn đang gọi.
- Nên tuân thủ điều khoản sử dụng của vbpl.vn khi crawl số lượng lớn.
