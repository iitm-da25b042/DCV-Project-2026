"""Convert the India TopoJSON (state layer) into a small GeoJSON
Source: https://github.com/udit-001/india-maps-data  (topojson/india.json, Census 2011 boundaries)
"""

import json
import sys


def decode_arcs(topo):
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:  # Arcs are delta-encoded integers
            x += dx
            y += dy
            pts.append([round(x * sx + tx, 3), round(y * sy + ty, 3)])
        arcs.append(pts)
    return arcs


def ring(arc_ids, arcs):
    out = []
    for i in arc_ids:
        pts = arcs[i] if i >= 0 else arcs[~i][::-1]
        out.extend(
            pts if not out else pts[1:]
        )  # Consecutive arcs share an endpoint
    return out


def convert(topo, layer="states"):
    arcs = decode_arcs(topo)
    feats = []
    for g in topo["objects"][layer]["geometries"]:
        if g["type"] == "Polygon":
            geom = {
                "type": "Polygon",
                "coordinates": [ring(r, arcs) for r in g["arcs"]],
            }
        elif g["type"] == "MultiPolygon":
            geom = {
                "type": "MultiPolygon",
                "coordinates": [
                    [ring(r, arcs) for r in poly] for poly in g["arcs"]
                ],
            }
        else:
            continue
        name = g["properties"]["st_nm"]
        feats.append(
            {
                "type": "Feature",
                "id": name,
                "properties": {"st_nm": name},
                "geometry": geom,
            }
        )
    return {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        topo = json.load(f)
    gj = convert(topo)
    with open(dst, "w") as f:
        json.dump(gj, f, separators=(",", ":"))
    print(f"Wrote {len(gj['features'])} state features to {dst}")
