from scripts.download_uk_caa_punctuality import discover_caa_csv_links


def test_discover_caa_csv_links_from_mocked_page(monkeypatch):
    class Response:
        text = '<a href="/Documents/Download/1">202401 Punctuality Statistics Full Analysis (CSV, 591 KB)</a>'
        def raise_for_status(self):
            pass
    def fake_get(url, timeout=30):
        return Response()
    monkeypatch.setattr("scripts.download_uk_caa_punctuality.requests.get", fake_get)
    links = discover_caa_csv_links(2024)
    assert len(links) == 1
    assert "202401" in links[0]["label"]
