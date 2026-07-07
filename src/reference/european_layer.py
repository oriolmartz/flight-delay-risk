"""Experimental European flight layer for FlightRisk.

Provides a curated catalog of major European airports and airlines plus
a lightweight adapter that auto-estimates route distance from airport
coordinates. The underlying ML model is still trained on BTS data, so this
layer should be presented as an experimental transfer layer for portfolio
demo purposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

EUROPEAN_AIRPORTS: dict[str, dict[str, float | str]] = {
    'BCN': {'name': 'Barcelona El Prat', 'city': 'Barcelona', 'country': 'Spain', 'lat': 41.2971, 'lon': 2.0785},
    'MAD': {'name': 'Adolfo Suárez Madrid-Barajas', 'city': 'Madrid', 'country': 'Spain', 'lat': 40.4893, 'lon': -3.5676},
    'LHR': {'name': 'London Heathrow', 'city': 'London', 'country': 'United Kingdom', 'lat': 51.47, 'lon': -0.4543},
    'LGW': {'name': 'London Gatwick', 'city': 'London', 'country': 'United Kingdom', 'lat': 51.1537, 'lon': -0.1821},
    'CDG': {'name': 'Paris Charles de Gaulle', 'city': 'Paris', 'country': 'France', 'lat': 49.0097, 'lon': 2.5479},
    'ORY': {'name': 'Paris Orly', 'city': 'Paris', 'country': 'France', 'lat': 48.7262, 'lon': 2.3652},
    'AMS': {'name': 'Amsterdam Schiphol', 'city': 'Amsterdam', 'country': 'Netherlands', 'lat': 52.3105, 'lon': 4.7683},
    'FRA': {'name': 'Frankfurt Airport', 'city': 'Frankfurt', 'country': 'Germany', 'lat': 50.0379, 'lon': 8.5622},
    'MUC': {'name': 'Munich Airport', 'city': 'Munich', 'country': 'Germany', 'lat': 48.3538, 'lon': 11.7861},
    'FCO': {'name': 'Rome Fiumicino', 'city': 'Rome', 'country': 'Italy', 'lat': 41.8003, 'lon': 12.2389},
    'MXP': {'name': 'Milan Malpensa', 'city': 'Milan', 'country': 'Italy', 'lat': 45.63, 'lon': 8.7281},
    'LIS': {'name': 'Lisbon Humberto Delgado', 'city': 'Lisbon', 'country': 'Portugal', 'lat': 38.7742, 'lon': -9.1342},
    'DUB': {'name': 'Dublin Airport', 'city': 'Dublin', 'country': 'Ireland', 'lat': 53.4213, 'lon': -6.2701},
    'CPH': {'name': 'Copenhagen Airport', 'city': 'Copenhagen', 'country': 'Denmark', 'lat': 55.6181, 'lon': 12.6560},
    'ARN': {'name': 'Stockholm Arlanda', 'city': 'Stockholm', 'country': 'Sweden', 'lat': 59.6519, 'lon': 17.9186},
    'ZRH': {'name': 'Zurich Airport', 'city': 'Zurich', 'country': 'Switzerland', 'lat': 47.4581, 'lon': 8.5555},
    'VIE': {'name': 'Vienna International', 'city': 'Vienna', 'country': 'Austria', 'lat': 48.1103, 'lon': 16.5697},
    'ATH': {'name': 'Athens International', 'city': 'Athens', 'country': 'Greece', 'lat': 37.9364, 'lon': 23.9445},
}

EUROPEAN_AIRLINES: dict[str, str] = {
    'IB': 'Iberia',
    'VY': 'Vueling',
    'BA': 'British Airways',
    'AF': 'Air France',
    'LH': 'Lufthansa',
    'KL': 'KLM',
    'FR': 'Ryanair',
    'U2': 'easyJet',
    'TP': 'TAP Air Portugal',
    'LX': 'Swiss',
    'OS': 'Austrian Airlines',
    'A3': 'Aegean Airlines',
    'SK': 'SAS',
    'EI': 'Aer Lingus',
    'AZ': 'ITA Airways',
}

@dataclass
class EuropeanRouteContext:
    airline: str
    airline_name: str
    origin: str
    destination: str
    origin_label: str
    destination_label: str
    distance_miles: float
    distance_source: str = 'estimated_from_airport_coordinates'
    region: str = 'europe_experimental'


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def european_airports_catalog() -> list[dict[str, str]]:
    items = []
    for code, data in sorted(EUROPEAN_AIRPORTS.items()):
        items.append({
            'code': code,
            'label': f"{code} · {data['city']} · {data['name']}",
            'country': str(data['country']),
        })
    return items


def european_airlines_catalog() -> list[dict[str, str]]:
    return [{'code': code, 'label': f'{code} · {name}'} for code, name in sorted(EUROPEAN_AIRLINES.items())]


def build_european_context(airline: str, origin: str, destination: str, distance: float | None = None) -> EuropeanRouteContext:
    airline = airline.strip().upper()
    origin = origin.strip().upper()
    destination = destination.strip().upper()
    if origin not in EUROPEAN_AIRPORTS:
        raise ValueError(f'Unsupported European origin airport: {origin}')
    if destination not in EUROPEAN_AIRPORTS:
        raise ValueError(f'Unsupported European destination airport: {destination}')
    if airline not in EUROPEAN_AIRLINES:
        raise ValueError(f'Unsupported European airline code: {airline}')

    o = EUROPEAN_AIRPORTS[origin]
    d = EUROPEAN_AIRPORTS[destination]
    if distance is None:
        distance = _haversine_miles(float(o['lat']), float(o['lon']), float(d['lat']), float(d['lon']))
        distance_source = 'estimated_from_airport_coordinates'
    else:
        distance_source = 'manual_input'
    return EuropeanRouteContext(
        airline=airline,
        airline_name=EUROPEAN_AIRLINES[airline],
        origin=origin,
        destination=destination,
        origin_label=f"{origin} · {o['city']} ({o['country']})",
        destination_label=f"{destination} · {d['city']} ({d['country']})",
        distance_miles=round(float(distance), 1),
        distance_source=distance_source,
    )
