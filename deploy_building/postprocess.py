"""
后处理: 把建筑物 mask GeoTIFF 矢量化
- 过滤小图斑(--min-area, 单位:像素数)
- 写出与栅格同 CRS 的多边形
- 三档兜底自动选驱动:
    1) fiona  (推荐, pip install fiona shapely 即可,自带 GDAL wheel)
    2) osgeo  (production 镜像 mindie-gdal-mmdet-mmseg 里有)
    3) 纯 Python 写 GeoJSON  (零依赖,前两个都没装时的最后保底)
- 输出文件名后缀 .shp -> 走 fiona/osgeo;后缀 .geojson -> 走纯 Python

用法:
    python postprocess.py --mask <mask.tif> --out <out.shp> [--min-area 25]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping


def _detect_backend():
    try:
        import fiona  # noqa: F401
        return 'fiona'
    except Exception:
        pass
    try:
        from osgeo import ogr, osr  # noqa: F401
        return 'osgeo'
    except Exception:
        pass
    return 'geojson'


def _write_shp_fiona(polys, areas_px, out_shp, crs_wkt):
    import fiona
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int', 'area_px': 'float'},
    }
    crs = None
    if crs_wkt:
        try:
            crs = fiona.crs.CRS.from_wkt(crs_wkt)
        except Exception:
            try:
                from fiona.crs import from_string
                crs = from_string(crs_wkt)
            except Exception:
                crs = None
    with fiona.open(str(out_shp), 'w', driver='ESRI Shapefile',
                    schema=schema, crs=crs, encoding='utf-8') as dst:
        for i, (p, a) in enumerate(zip(polys, areas_px)):
            dst.write({
                'geometry': mapping(p),
                'properties': {'id': i + 1, 'area_px': float(a)},
            })


def _write_shp_osgeo(polys, areas_px, out_shp, crs_wkt):
    from osgeo import ogr, osr
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        p = str(out_shp)[:-4] + ext
        if os.path.exists(p):
            os.remove(p)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    ds = drv.CreateDataSource(str(out_shp))
    srs = osr.SpatialReference()
    if crs_wkt:
        srs.ImportFromWkt(crs_wkt)
    layer = ds.CreateLayer('building', srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('area_px', ogr.OFTReal))
    for i, (p, a) in enumerate(zip(polys, areas_px)):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField('id', i + 1)
        feat.SetField('area_px', float(a))
        feat.SetGeometry(ogr.CreateGeometryFromWkb(p.wkb))
        layer.CreateFeature(feat)
        feat = None
    ds = None


def _write_geojson(polys, areas_px, out_geojson, crs_wkt):
    features = []
    for i, (p, a) in enumerate(zip(polys, areas_px)):
        features.append({
            'type': 'Feature',
            'id': i + 1,
            'properties': {'id': i + 1, 'area_px': float(a)},
            'geometry': mapping(p),
        })
    fc = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': (crs_wkt[:300] + '...') if len(crs_wkt) > 300 else crs_wkt},
        },
        'features': features,
    }
    with open(out_geojson, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False)


def vectorize(mask_path, out_path, min_area_px=25):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    print(f'[vec] mask shape {mask.shape}, building px = {(mask>0).sum()}', flush=True)

    raw = []
    for geom, val in shapes(mask, mask=(mask > 0), transform=transform):
        if val == 0:
            continue
        raw.append(geom)
    print(f'[vec] 原始多边形: {len(raw)}', flush=True)

    px_w = abs(transform.a)
    px_h = abs(transform.e)
    px_area = px_w * px_h
    min_area_crs = min_area_px * px_area
    polys = [shape(g) for g in raw]
    keep = [(p, p.area / px_area) for p in polys if p.area >= min_area_crs]
    print(f'[vec] 过滤后(>= {min_area_px} 像元): {len(keep)}', flush=True)

    if not keep:
        print('[vec] 没有多边形可写,跳过', flush=True)
        return None

    polys_keep, areas_keep = zip(*keep)
    crs_wkt = crs.to_wkt() if crs else ''

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    backend = _detect_backend()
    print(f'[vec] backend = {backend}', flush=True)

    if out_p.suffix.lower() == '.shp':
        if backend == 'fiona':
            _write_shp_fiona(polys_keep, areas_keep, out_p, crs_wkt)
            print(f'[ok] shp (fiona) -> {out_p}  ({len(polys_keep)} polygons)', flush=True)
        elif backend == 'osgeo':
            _write_shp_osgeo(polys_keep, areas_keep, out_p, crs_wkt)
            print(f'[ok] shp (osgeo) -> {out_p}  ({len(polys_keep)} polygons)', flush=True)
        else:
            geo = out_p.with_suffix('.geojson')
            _write_geojson(polys_keep, areas_keep, geo, crs_wkt)
            print(f'[warn] fiona/osgeo 都不可用,改写 GeoJSON: {geo}', flush=True)
            print(f'       pip install fiona  或在 production GDAL 镜像里就能直接出 .shp', flush=True)
            out_p = geo
    else:
        _write_geojson(polys_keep, areas_keep, out_p, crs_wkt)
        print(f'[ok] geojson -> {out_p}  ({len(polys_keep)} polygons)', flush=True)

    return str(out_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', required=True, help='建筑物 mask GeoTIFF')
    ap.add_argument('--out', required=True, help='输出 .shp 或 .geojson')
    ap.add_argument('--min-area', type=int, default=25,
                    help='最小图斑像素数,小于此值被过滤(默认 25)')
    args = ap.parse_args()
    vectorize(args.mask, args.out, args.min_area)


if __name__ == '__main__':
    main()
