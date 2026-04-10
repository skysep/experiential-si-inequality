import time
import math
import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Iterable
import requests

import warnings
warnings.filterwarnings('ignore')

cm = 1/2.54
palette = ["#3F3517",  '#CE2E31', '#C96F6B', '#CCA464', '#F8D768', '#F0DCD4']


def _request_body(
    included_types: List[str],
    page_size: int = 20,
    circle_center: Optional[Tuple[float, float]] = None,
    circle_radius_m: Optional[int] = None,
    routing_origin: Optional[Tuple[float, float]] = None,
    travel_mode: Optional[str] = None #"DRIVE"
) -> Dict:
    body: Dict = {
        "includedTypes": included_types,
        "maxResultCount": page_size
    }

    if circle_center and circle_radius_m:
        body["locationRestriction"] = {
            "circle": {
                "center": {"latitude": circle_center[0], "longitude": circle_center[1]},
                "radius": circle_radius_m
            }
        }
    else:
        raise ValueError("Provide circle_center+circle_radius_m.")

    if routing_origin:
        body["routingParameters"] = {
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": routing_origin[0],
                        "longitude": routing_origin[1]
                    }
                }
            },
            "travelMode": travel_mode
        }

    return body


def _field_mask(include_routing: bool) -> str:
    base = [
        "places.id",
        "places.displayName",
        "places.types",
        "places.location",
        "places.addressDescriptor",
        # "places.formattedAddress",
        # "places.googleMapsUri",

        "places.accessibilityOptions",
        "places.attributions",
        # "places.currentOpeningHours",
        "places.regularOpeningHours",
        "places.priceLevel",
        "places.priceRange",
        "places.rating",
        "places.userRatingCount",

        "places.parkingOptions",
        "places.restroom",

    ]
    if include_routing:
        base.append("routingSummaries")
    return ",".join(base)



def _headers(api_key: str, include_routing: bool) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": _field_mask(include_routing)
    }


def _backoff_sleep(retry_index: int) -> None:
    base = max(60, (2 ** retry_index))
    time.sleep(base + (0.25 * math.sin(time.time())))


def search_nearby_places(
    api_key: str,
    places_end_point: str,
    included_types: List[str],
    page_size: int = 20,
    circle_center: Optional[Tuple[float, float]] = None,
    circle_radius_m: Optional[int] = None,
    routing_origin: Optional[Tuple[float, float]] = None,
    travel_mode: Optional[str] =  None, #"DRIVE"
) -> Iterable[Dict]:
    """
    Yields dicts for each place result across all pages.
    If routing_origin is provided, response includes routingSummaries for the page.
    """
    include_routing = routing_origin is not None
    headers = _headers(api_key, include_routing)
    body = _request_body(
        included_types=included_types,
        page_size=page_size,
        circle_center=circle_center,
        circle_radius_m=circle_radius_m,
        routing_origin=routing_origin,
        travel_mode=travel_mode
    )


    while True:
        for attempt in range(6):
            resp = requests.post(places_end_point, headers=headers, json=body, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                break

            if resp.status_code in (429, 500, 502, 503, 504):
                _backoff_sleep(attempt)
                continue
            raise RuntimeError(f"Nearby Search error {resp.status_code}: {resp.text}")

        places = data.get("places", [])
        
        time.sleep(0.5)
        return places


