from src.reference.european_layer import build_european_context, european_airports_catalog


def test_build_european_context_estimates_distance():
    ctx = build_european_context('IB', 'BCN', 'AMS')
    assert ctx.distance_miles > 0
    assert ctx.origin == 'BCN'
    assert ctx.destination == 'AMS'


def test_airport_catalog_has_labels():
    airports = european_airports_catalog()
    assert any(item['code'] == 'BCN' for item in airports)
    assert all('label' in item for item in airports)
