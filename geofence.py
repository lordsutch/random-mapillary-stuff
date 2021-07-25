#!/usr/bin/env python3

import argparse
import datetime
import math
import os
import shlex
import sys
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import cache
from pathlib import Path

import astral
import astral.sun
import exif
import pyproj
import shapely.geometry
import shapely.ops
from progress.bar import Bar

WGS84 = pyproj.CRS('epsg:4326')


# convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
# tweaked a bit - CNL
def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = (math.floor((lon + 180) / 6) % 60) + 1
    code = (32600 + utm_band) if lat >= 0 else (32700 + utm_band)
    return code


class Geofence:
    def __init__(self, *shapes):
        self.shapes = list(shapes)
    
    def __contains__(self, point: shapely.geometry.Point) -> bool:
        # for x in self.shapes:
        #     # print(point, x.centroid, point.within(x))
        #     if x.contains(point):
        #         return True
            
        return any(x.contains(point) for x in self.shapes)

    def add_shape(self, shape):
        self.shapes.append(shape)

    def __repr__(self):
        return 'Geofence('+','.join(repr(shape) for shape in self.shapes)+')'
    

def dms_to_decimal(coords: tuple, hemisphere: str) -> float:
    '''Convert a DMS tuple to decimal degrees.'''
    deccoord = coords[0] + coords[1]/60 + coords[2]/3600
    return deccoord if hemisphere in 'NE' else -deccoord


def parse_geofence_spec(spec: str, infile: Path) -> Geofence:
    fence = Geofence()
    lexer = shlex.shlex(spec, infile, posix=True, punctuation_chars=' ')
    
    while token := lexer.get_token():
        if token == 'point':
            lat = float(lexer.get_token())
            lon = float(lexer.get_token())
            radius = float(lexer.get_token())

            epsg_number = convert_wgs_to_utm(lon, lat)
            p = pyproj.Proj(f'epsg:{epsg_number}')
            x, y = p(lon, lat)
            utm_point = shapely.geometry.Point(x, y)
            utm_buffered_point = utm_point.buffer(radius)
            projector = pyproj.Transformer.from_crs(p.crs, WGS84,
                                                    always_xy=True).transform
            buffered_point = shapely.ops.transform(projector,
                                                   utm_buffered_point)
            fence.add_shape(buffered_point)
        elif token == 'bbox':
            lat1, lon1 = float(lexer.get_token()), float(lexer.get_token())
            lat2, lon2 = float(lexer.get_token()), float(lexer.get_token())
            
            fence.add_shape(shapely.geometry.Polygon(
                ((lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2))))
        elif token == 'poly':
            shape = []
            while (latstr := lexer.get_token()) != 'endpoly':
                lat, lon = float(latstr), float(lexer.get_token())
                shape.append((lat, lon))
            if len(shape) < 3:
                print(f'Polygon should have >= 3 sides, got {len(shape)}.',
                      file=sys.stderr)
                continue
            fence.add_shape(shapely.geometry.Polygon(shape))
        else:
            print(f'{lexer.error_leader()}Unknown token {token}',
                  file=sys.stderr)
    
    return fence


def geofence_image(fence: Geofence, lat: float, lon: float,
                   imgtime: datetime.datetime = None,
                   filter_dark=True) -> bool:
    '''Returns True if we should keep the image, False otherwise.'''
    filtered = False
    tz = datetime.timezone(datetime.timedelta(hours=(lon+7.5) // 15))
    
    if filter_dark:
        date = imgtime.astimezone(tz).date()
        loc = astral.Observer(latitude=lat, longitude=lon)
        try:
            sunrise, sunset = astral.sun.daylight(loc, date, tz)
            # print(date, imgtime, sunrise, sunset)
            if imgtime < sunrise or imgtime > sunset:
                filtered = True
        except ValueError:
            noon = astral.sun.noon(loc, date=date)
            if astral.sun.elevation(noon) < 0:
                # Sun is not up at noon, so it's winter
                filtered = True
    
    position = shapely.geometry.Point(lon, lat)

    return (filtered or position in fence)


def geofence_file(imgfile: os.PathLike, fence: Geofence,
                  save_as: os.PathLike = None,
                  move_to: os.PathLike = None,
                  filter_dark=True,
                  strip_gps=False):
    with open(imgfile, 'rb') as imfp:
        my_image = exif.Image(imfp)

    if not my_image.has_exif:
        return

    lat = my_image.get('gps_latitude', None)
    lon = my_image.get('gps_longitude', None)
    if lat is None or lon is None:
        return

    latref = my_image.get('gps_latitude_ref', '')
    lonref = my_image.get('gps_longitude_ref', '')
    if not latref or not lonref:
        return

    dlon, dlat = dms_to_decimal(lon, lonref), dms_to_decimal(lat, latref)

    dto = my_image.get('datetime_original', '')
    imgtime = datetime.datetime.strptime(dto, '%Y:%m:%d %H:%M:%S').replace(
        tzinfo=datetime.timezone.utc)
    
    if not geofence_image(fence, dlat, dlon, imgtime, filter_dark):
        return
    
    if strip_gps:
        for tag in my_image.list_all():
            if tag.startswith('gps_'):
                my_image.delete(tag)
        if move_to:
            outpath = Path(move_to) / (save_as or imgfile).name
        else:
            outpath = (save_as or imgfile)

        with open(outpath, 'wb') as imfp:
            imfp.write(my_image.get_file())

        if move_to:
            imgfile.unlink()
        return outpath
    elif move_to:
        new_filename = Path(move_to) / (save_as or imgfile).name
        imgfile.rename(new_filename)
        return new_filename
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='handle images within geographic boundaries differently')
    parser.add_argument('--strip-gps', action='store_true',
                        help='remove geotagging from matching files')
    parser.add_argument('--move', type=str, metavar='DIR',
                        help='move matching files to DIR')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='descend into subdirectories too')
    parser.add_argument('--fence', action='store', type=Path,
                        default=Path('geofence-spec'),
                        help='file to read geofences from')
    parser.add_argument('--keep-night-photos', action='store_true',
                        help="don't filter out photos taken before "
                        "sunrise/after sunset")
    parser.add_argument('files', type=Path, nargs='+', metavar='FILE|DIR',
                        help='files and/or directories to process')
    args = parser.parse_args()

    to_process = []
    pathname: Path
    for pathname in args.files:
        if pathname.is_file():
            to_process.append(pathname)
        elif pathname.is_dir():
            if args.recursive:
                to_process.extend(path for path in pathname.rglob('*')
                                  if path.is_file())
            else:
                to_process.extend(path for path in pathname.iterdir()
                                  if path.is_file())

    args.fence: Path
    if args.fence:
        if not args.fence.is_file():
            print(args.fence, 'not found', file=sys.stderr)
            sys.exit(1)
            
        fence = parse_geofence_spec(args.fence.read_text(), args.fence)
        if not fence:
            print(f'No fence information found in {args.fence}, aborting.')
            sys.exit(1)

    filtered = []
    for path in Bar('Geofencing').iter(to_process):
        filtered_file = geofence_file(path, fence, move_to=args.move,
                                      strip_gps=args.strip_gps,
                                      filter_dark=not args.keep_night_photos)
        if filtered_file:
            filtered.append(filtered_file)
            
    if len(filtered):
        print(f'Filtered {len(filtered)} files.')
