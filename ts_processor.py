#For this: https://forum.mapillary.com/t/blueskysea-b4k-viofo-a119v3-and-mapillary
#Usage: python ts_processor.py --input 20210311080720_000421.TS  --sampling_interval 0.5 --folder output

import argparse
import bz2
import cgitb
import concurrent.futures
import configparser
import contextlib
import glob
import gzip
import inspect
import io
import lzma
import logging
import math
import mmap
import multiprocessing
import os
import re
import struct
import sys
import webbrowser
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from fractions import Fraction
from functools import cache
from itertools import groupby, accumulate
from pathlib import Path
from typing import Optional, List, Callable, TextIO, Union, Iterable

import av
import av.filter
import gpmf
import gpxpy
import piexif
import pyproj
from PIL import Image
from pymp4.parser import Box

import geofence
from gpx_interpolate import gpx_interpolate, gpx_calculate_speed

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# Match filenames ending in mp4 or ts that don't start with a period
VALID_FILENAME = re.compile(r'^(?!\.).*\.(mp4|ts)$', re.IGNORECASE)

WGS84 = pyproj.Geod(ellps='WGS84')
Mercator = pyproj.Proj(proj='webmerc', datum='WGS84')

gps_struct = struct.Struct('<I I I I I I ccc x f f f f')
KNOTS_TO_MPS = 1852.0 / 3600.0

DATETIME_STR_FORMAT = '%Y:%m:%d %H:%M:%S'
DATE_STR_FORMAT = '%Y:%m:%d'

THUMB_SIZE = 200
THUMB_QUALITY = 'web_low'

PIL_SAVE_SETTINGS = dict(quality=85,
                         # 4:2:0;  Source subsampling isn't better
                         subsampling=2,
                         # optimize=True,
                         progressive=True)

EXTENSIONS = {'jpeg': 'jpg'}

# JPEG_SETTINGS = (cv2.IMWRITE_JPEG_OPTIMIZE, 1,
#                  cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
#                  cv2.IMWRITE_JPEG_QUALITY, 90,
#                  # cv2.IMWRITE_JPEG_CHROMA_QUALITY, 80,
#                  )

logger = multiprocessing.log_to_stderr()

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object): #from here: https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0] ,1)
        os.dup2(self.null_fds[1] ,2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0] ,1)
        os.dup2(self.save_fds[1] ,2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def to_deg(value: float, loc: Iterable):  #From here: https://gist.github.com/c060604
    """convert decimal coordinates into degrees, munutes and seconds tuple
    Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
    return: tuple like (25, 13, 48.343 ,'N')
    """
    loc_value = loc[int(value >= 0)]

    degrees, minutes = divmod(abs(value), 1)
    minutes, seconds = divmod(minutes*60, 1)

    return (int(degrees), int(minutes), seconds*60, loc_value)


# Stolen from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr, hr


def change_to_rational(number):
    """convert a number to rational
    Keyword arguments: number
    return: tuple like (1, 2), (numerator, denominator)
    """
    f = Fraction(number).limit_denominator()
    return (f.numerator, f.denominator)


EXIF_MAPPER = {
    'max aperture': (piexif.ExifIFD.MaxApertureValue,
                     change_to_rational),
    'focal length': (piexif.ExifIFD.FocalLength, change_to_rational),
    '35mm equivalent focal length': (piexif.ExifIFD.FocalLengthIn35mmFilm,
                                     int),
    'focal plane x resolution': (piexif.ExifIFD.FocalPlaneXResolution,
                                 change_to_rational),
    'focal plane y resolution': (piexif.ExifIFD.FocalPlaneYResolution,
                                 change_to_rational),
}

# A129 Pro
# 7.2mm diagonal - crop factor of ~6 - 6.28 x 3.53mm
# Pixel size 1.62 µm x 1.62 µm


def build_exif_data(position: gpxpy.gpx.GPXTrackPoint, make: str, model: str,
                    width: int = None,
                    height: int = None, exifdata: dict = None,
                    thumbnail: bytearray = None, bearing_modifier: float = 0):
    """Adds GPS position as EXIF metadata
    """
    if not exifdata:
        exifdata = {}

    lat_deg = to_deg(round(position.latitude, 7), ("S", "N"))
    lng_deg = to_deg(round(position.longitude, 7), ("W", "E"))

    exiv_lat = tuple(change_to_rational(x) for x in lat_deg[:3])
    exiv_lng = tuple(change_to_rational(x) for x in lng_deg[:3])
    bearing = round((position.course + bearing_modifier) % 360, 2)
    nbear = change_to_rational(bearing)

    utctime = position.time.astimezone(timezone.utc)
    utcdatestamp = utctime.strftime(DATE_STR_FORMAT)
    fracsecs = utctime.second + Fraction(utctime.microsecond, 1_000_000)
    utctimestamp = (change_to_rational(utctime.hour),
                    change_to_rational(utctime.minute),
                    change_to_rational(fracsecs))

    dtstr = position.time.strftime(DATETIME_STR_FORMAT)
    subsecstr = f'{(position.time.microsecond / 1000):03.0f}'

    if position.type_of_gpx_fix is None:
        measuremode = str(2+(position.elevation is not None))
    elif position.type_of_gpx_fix == '2d':
        measuremode = '2'
    else:
        measuremode = '3'

    dop = (position.horizontal_dilution if measuremode == '2'
           else position.position_dilution)

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
        piexif.GPSIFD.GPSLatitude: exiv_lat,
        piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
        piexif.GPSIFD.GPSLongitude: exiv_lng,
        piexif.GPSIFD.GPSImgDirection: nbear,
        piexif.GPSIFD.GPSImgDirectionRef: 'T',
        piexif.GPSIFD.GPSMapDatum: 'WGS84',
        piexif.GPSIFD.GPSDateStamp: utcdatestamp,
        piexif.GPSIFD.GPSTimeStamp: utctimestamp,
        piexif.GPSIFD.GPSMeasureMode: measuremode,
    }

    if position.speed is not None:
        gps_ifd[piexif.GPSIFD.GPSSpeed] = change_to_rational(
            round(position.speed*3.6, 1))
        gps_ifd[piexif.GPSIFD.GPSSpeedRef] = 'K'

    if position.elevation is not None:
        gps_ifd[piexif.GPSIFD.GPSAltitude] = change_to_rational(
            round(abs(position.elevation), 1))
        gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = int(position.elevation < 0)

    if position.satellites is not None:
        gps_ifd[piexif.GPSIFD.GPSSatellites] = str(position.satellites)

    if dop:
        gps_ifd[piexif.GPSIFD.GPSDOP] = change_to_rational(round(dop, 2))

    zeroth_ifd = {
        piexif.ImageIFD.Make: make,
        piexif.ImageIFD.Model: model,
    }

    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: dtstr,
        piexif.ExifIFD.SubSecTimeOriginal: subsecstr,
        piexif.ExifIFD.SceneType: b'\x01',       # Directly photographed
        piexif.ExifIFD.FileSource: b'\x03',      # Digital camera
        piexif.ExifIFD.SubjectDistanceRange: 3,  # Distant subject
    }

    if width:
        zeroth_ifd[piexif.ImageIFD.ImageWidth] = width
        exif_ifd[piexif.ExifIFD.PixelXDimension] = width

    if height:
        zeroth_ifd[piexif.ImageIFD.ImageLength] = height
        exif_ifd[piexif.ExifIFD.PixelYDimension] = height

    for k, v in exifdata.items():
        attribute, func = EXIF_MAPPER.get(k, (None, None))
        if attribute:
            res = func(v) if func else v
            exif_ifd[attribute] = res
            if attribute in (piexif.ExifIFD.FocalPlaneXResolution,
                             piexif.ExifIFD.FocalPlaneYResolution):
                # Only store resolution unit if necessary
                exif_ifd[piexif.ExifIFD.FocalPlaneResolutionUnit] = 3

    # print(exif_ifd)
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd}

    if thumbnail:
        exif_dict['1st'] = {}
        exif_dict['thumbnail'] = thumbnail

    exif_bytes = piexif.dump(exif_dict)
    return exif_bytes


def set_gps_location(file_name, **args):
    exif_bytes = build_exif_data(**args)
    piexif.insert(exif_bytes, file_name)


def fix_coordinates(hemisphere: bytes, coordinate: float) -> float: #From here: https://sergei.nz/extracting-gps-data-from-viofo-a119-and-other-novatek-powered-cameras/
    # coordinate, = coordinate_input
    degrees, minutes = divmod(float(coordinate), 100)
    return (degrees + minutes/60) * (-1.0 if hemisphere in b'SW' else 1.0)


def to_gps_latlon(v, refs):
    ref = refs[0] if v >= 0 else refs[1]
    dd = abs(v)
    d = int(dd)
    mm = (dd - d) * 60
    m = int(mm)
    ss = (mm - m) * 60
    # s = int(ss * 100)
    r = (d, m, ss)
    return (ref, r)


# @cache
def lonlat_metric(xlon, xlat):
    return Mercator(xlon, xlat)


# @cache
def metric_lonlat(xmx, ymy):
    return Mercator(xmx, ymy, inverse=True)


def decode_novatek_gps_packet(packet: bytes, tz: tzinfo=timezone.utc,
                              logger=logging) -> Optional[dict]:
    # Different units and firmware seem to put the data in different places
    if m := re.search(rb'[ AV][0NS][0EW]', packet):
        offset = m.start()
        input_packet = packet[offset-24:]
    else:
        logger.debug('Unknown packet: %s', packet[:188])
        logger.debug(packet[:188].hex(' ', 8))
        return None

    unpacked = gps_struct.unpack_from(input_packet)
    (hour, minute, second, year, month, day,
     active, lathem, lonhem, enclat, enclon, speed_knots,
     bearing) = unpacked

    if active == b'A':
        # Decode Novatek coordinate format to decimal degrees
        lat = fix_coordinates(lathem, enclat)
        lon = fix_coordinates(lonhem, enclon)
        # mx, my = lonlat_metric(lon, lat)
    else:
        lat = lon = None
        # mx = my = None

    # Validity checks
    if (not (0 <= hour <= 23) or not (0 <= minute <= 59) or
        not (0 <= second <= 60) or not (1 <= month <= 12) or
        not (1 <= day <= 31) or not (0 <= bearing <= 360) or
        speed_knots < 0 or active not in b'AV'):
        # Packet is invalid
        logger.debug('Packet invalid: %s', unpacked)
        return None

    ts = datetime(year=2000+year, month=month, day=day, hour=hour,
                  minute=minute, second=second, tzinfo=tz)
    if active == b'V':
        # Time-only fix
        return dict(ts=ts, active=active, fix='none')
    elif active != b'A':
        logger.debug('No GPS fix: %s', unpacked)
        return None

    speed = speed_knots * KNOTS_TO_MPS
    return dict(lat=lat, latR=lathem, lon=lon, lonR=lonhem, ts=ts,
                bearing=bearing, speed=speed, active=active, fix='2d')


def detect_file_type(input_file, device_override='', logger=logging):
    device = "X"
    make = "unknown"
    model = "unknown"
    detected_offsets = []

    p = Path(input_file)
    extension = p.suffix.lower()
    known = extension in ('.ts', '.mp4')  # Try everything if extension unknown

    if extension == '.ts' or not known:
        with open(input_file, "rb") as f:
            device = "A"
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.madvise(mmap.MADV_SEQUENTIAL)
                input_packet = mm.read(188)
                if b"\xB0\x0D\x30\x34\xC3" in input_packet[4:20] or device_override == "V":
                    device = "V"
                    make = "Viofo"
                    model = "A119 V3"
                elif b"\xB0\x0D\x30\x34\xC3" in input_packet[20:40] or device_override == "S":
                    device = "S"
                    make = "Viofo"
                    model = "A119S"
                elif b"\x40\x1F\x4E\x54\x39" in input_packet[4:20] or device_override == "B":
                    device = "B"
                    make = "Blueskysea"
                    model = "B4K"

                while device == "A":
                    input_packet = mm.read(188)
                    if not input_packet:
                        break

                    #Autodetect camera type
                    if input_packet.startswith(b"\x47\x03\x00") and decode_novatek_gps_packet(input_packet):
                        device = "B"
                        make = "Blueskysea"
                        model = "B4K"
                        logger.debug("Autodetected as Blueskysea B4K")
                        break
                    elif input_packet.startswith(b"\x47\x43\x00") and decode_novatek_gps_packet(input_packet):
                        logger.debug("Autodetected as Viofo A119 V3")
                        make = "Viofo"
                        model = "A119 V3"
                        break

    if device == 'X' and extension == '.mp4' or not known:
        # Guess which MP4 method is used: Novatek, Subtitle, NMEA
        try:
            container = av.open(str(input_file), 'r')
        except av.error.InvalidDataError:
            container = None
        if container and len(container.streams) >= 4:
            # GoPro would have >= 4 streams
            candidate = container.streams[3]
            if candidate.type == 'data':
                hname = candidate.metadata.get('handler_name', '')
                if 'GoPro MET' in hname:
                    container.close()
                    return 'P', 'GoPro', 'HERO', []
        if container:
            container.close()
        with open(input_file, "rb") as fx:
            with mmap.mmap(fx.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.madvise(mmap.MADV_RANDOM)
                eof = mm.size()
                while mm.tell() < eof:
                    lazybox = struct.unpack('!I 4s',
                                            mm[mm.tell():mm.tell()+8])
                    # Save some parsing overhead since we don't care...
                    if lazybox[1] in (b'ftyp', b'mdat', b'skip') and lazybox[0] > 1:
                        try:
                            mm.seek(lazybox[0], os.SEEK_CUR)
                        except ValueError:
                            mm.seek(0, os.SEEK_END)
                        continue

                    try:
                        box = Box.parse_stream(mm)
                    except:
                        continue
                    # print(box.type.decode("utf-8"))
                    if box.type == b"free":
                        length = len(box.data)
                        offset = 0
                        while offset < length:
                            inp = Box.parse(box.data[offset:])
                            #print (inp.type.decode("utf-8"))
                            if inp.type == b"gps": #NMEA-based
                                lines = inp.data
                                for line in lines.splitlines():
                                    # m = str(line).lstrip("[]0123456789")
                                    if b"$GPGGA" in line:
                                        device = "N"
                                        make = "NMEA-based video"
                                        model = "unknown"
                                        break
                            offset += inp.end
                    elif box.type == b"gps" and len(box.data) >= 16: #has Novatek-specific stuff
                        offset, length = struct.unpack_from('>8x II', box.data)
                        # largeelem = fx.read()
                        # print(f'{offset:#8x} {length:#8x}')
                        if mm[offset:offset+length].find(b'freeGPS') >= 0:
                            make = "Novatek"
                            model = "MP4"
                            device = "T"
                            for offset, length in struct.iter_unpack(
                                    '>II', box.data[8:]):
                                if offset == 0 or length == 0:
                                    break
                                detected_offsets.append((offset, length))

                    elif box.type == b"moov":
                        try:
                            length = len(box.data)
                        except:
                            length = 0
                        offset = 0
                        while offset < length:
                            inp = Box.parse(box.data[offset:])
                            if inp.type == b"gps": #NMEA-based
                                for line in lines.splitlines():
                                    if b"$GPGGA" in line:
                                        device = "N"
                                        make = "NMEA-based video"
                                        model = "unknown"
                                        break

                            offset += inp.end

                if device == "X":
                    mm.seek(0)
                    if mm.find(b'\x00\x14\x50\x4E\x44\x4D\x00\x00\x00\x00') >= 0:
                        make = "Garmin"
                        model = "unknown"
                        device = "G"

    return device, make, model, detected_offsets


def get_gps_data_nt(input_ts_file, device, tz, logger=logging,
                    offsets=None) -> gpxpy.gpx.GPXTrackSegment:
    packetno = 0
    locdata = {}
    with open(input_ts_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_RANDOM)
            for (offset, length) in offsets:
                input_packet = mm[offset:offset+length]
                pos = input_packet.find(b'freeGPS')
                if pos < 0:
                    # print('?bogus packet at', offset)
                    continue
                else:
                    input_packet = input_packet[pos:]

                if currentdata := decode_novatek_gps_packet(
                        input_packet, tz, logger):
                    locdata[packetno] = currentdata
                    packetno += 1
            if packetno > 0:
                return locdata_to_gpxsegment(locdata)

            # Slow scan
            logger.debug('no GPS data found fast way; attempting full scan')
            mm.madvise(mmap.MADV_SEQUENTIAL)
            # If it's a real packet there should be a N or S followed by E or W
            for match in re.finditer(
                    b'(freeGPS).{1,999}[0NS][0EW]', mm[0:]):
                logger.debug('packet at %d-%d', match.start(), match.end())
                input_packet = mm[match.start():match.end()+50]
                if currentdata := decode_novatek_gps_packet(
                        input_packet, tz, logger):
                    locdata[packetno] = currentdata
                    packetno += 1

    return locdata_to_gpxsegment(locdata)


garmin_gps_struct = struct.Struct('>s xx i i')


def get_gps_data_garmin(input_ts_file, device,
                        tz) -> gpxpy.gpx.GPXTrackSegment:
    packetno = 0
    locdata = {}

    with open(input_ts_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            for match in re.finditer(b'\x00\x14\x50\x4E\x44\x4D\x00\x00\x00\x00', mm):
                input_packet = mm[match.start():match.start()+56]
                speed_kmh, lat, lon = garmin_gps_struct.unpack_from(input_packet)
                if lat and lon:
                    lat /= 11930464.711111112
                    lon /= 11930464.711111112
                else:
                    lat = lon = None

                locdata[packetno] = dict(
                    speed=speed_kmh / 3.6,
                    bearing=0, ts=datetime.fromtimestamp(0, timezone.utc),
                    posix_clock=0, active=b'A',
                    lat=lat, lon=lon,
                    latR=b'N' if lat >= 0 else b'S',
                    lonR=b'E' if lon >= 0 else b'W',
                    fix='2d')
                packetno += 1

    return locdata_to_gpxsegment(locdata)


def parse_nmea_rmc(line: bytes, tzone='Z') -> dict:
    bits = line.split(b',')
    datestr = f'{bits[9].decode("us-ascii")} {bits[1].decode("us-ascii")}{tzone}'
    ts = datetime.strptime(datestr, '%d%m%y %H%M%S.%f%z')

    rawlat, lathem, rawlon, lonhem = bits[3:7]
    lat = fix_coordinates(lathem, rawlat) if rawlat else None
    lon = fix_coordinates(lonhem, rawlon) if rawlon else None

    return dict(ts=ts, posix_clock=ts.timestamp(),
                active=bits[2], lat=lat, latR=lathem, lon=lon,
                lonR=lonhem, fix='2d',
                speed=float(bits[7])*KNOTS_TO_MPS,
                bearing=float(bits[8]), metric=0, prevdist=0)


def get_gps_data_nmea(input_file, device, tz) -> gpxpy.gpx.GPXTrackSegment:
    packetno = 0
    locdata = {}

    offset = tz.utcoffset().total_seconds()
    sign = '+' if offset >= 0 else '-'
    minutes, seconds = divmod(abs(offset), 60)
    hours, minutes = divmod(minutes, 60)
    tzstr = f'{sign}{hours:02d}{minutes:02d}'

    with open(input_file, "rb") as fx:
        with mmap.mmap(fx.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            eof = mm.size()
            prevts = datetime.fromtimestamp(0, timezone.utc)
            while mm.tell() < eof:
                try:
                    box = Box.parse_stream(fx)
                except:
                    pass
                if box.type == b"free":
                    try:
                        length = len(box.data)
                    except:
                        length = 0
                    offset = 0
                    while offset < length:
                        inp = Box.parse(box.data[offset:])
                        # print (inp.type.decode("utf-8"))
                        if inp.type == b"gps":  # NMEA-based
                            lines = inp.data
                            for line in lines.splitlines():
                                if b"$GPRMC" in line:
                                    currentdata = parse_nmea_rmc(line, tzstr)
                                    ts = currentdata['ts']
                                    active = currentdata['active']
                                    if ts > prevts:
                                        locdata[packetno] = currentdata
                                        prevts = ts
                                        packetno += 1
                        offset += inp.end

    return locdata_to_gpxsegment(locdata)


def get_gps_data_ts(input_ts_file, device, tz,
                    logger=logging) -> gpxpy.gpx.GPXTrackSegment:
    packetno = 0
    locdata = {}
    prevdata = {}
    with open(input_ts_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            input_packet = mm.read(188)  # First packet, try to autodetect

            while True:
                input_packet = mm.read(188)
                if not input_packet:
                    break

                currentdata = {}
                if device == 'B' and prevdata and input_packet.startswith(b"\x47\x03\x00"):
                    if currentdata := decode_novatek_gps_packet(input_packet, tz, logger):
                        # Combine data from the two packets
                        subdict = dict(hour=prevdata['hour'],
                                       minute=prevdata['minute'],
                                       second=prevdata['second'],
                                       year=prevdata['year'],
                                       month=prevdata['month'],
                                       day=prevdata['day'])
                        packetdata, prevdata = currentdata.copy(), currentdata
                        packetdata.update(subdict)
                        locdata[packetno] = packetdata
                        packetno += 1
                elif device in ('V', 'S') and input_packet.startswith(b"\x47\x43\x00"):
                    if currentdata := decode_novatek_gps_packet(input_packet, tz, logger):
                        locdata[packetno] = currentdata
                        prevdata = currentdata
                    packetno += 1

    return locdata, packetno


def get_gps_data_gopro(input_ts_file: os.PathLike, device: str,
                       tzone) -> gpxpy.gpx.GPXTrackSegment:
    container = av.open(str(input_ts_file), 'r')
    try:
        gpmf_stream = container.streams[3]
    except IndexError:
        container.close()
        return gpxpy.gpx.GPXTrackSegment()

    if 'GoPro MET' not in (hname :=
                           gpmf_stream.metadata.get('handler_name', '')):
        container.close()
        return gpxpy.gpx.GPXTrackSegment()

    # There should be raw GPMF data if we demux
    content = b''
    for packet in container.demux(streams=3):
        content += bytes(packet)
    container.close()

    gps_blocks = gpmf.gps.extract_gps_blocks(content)
    gps_data = [gpmf.gps.parse_gps_block(x) for x in gps_blocks]

    # Parse to GPX format
    segment = gpmf.gps.make_pgx_segment(gps_data)

    # Convert times to tzone
    for point in segment.points:
        point.time = point.time.replace(tzinfo=tzone)

    return segment


# Borked
def build_gpxtree(locdata: List[tuple], make: str, model: str,
                  generate_track=True) -> ET.ElementTree:
    gpx = ET.Element('gpx', version='1.0', creator='ts_processor')
    desc = ET.SubElement(gpx, 'desc')
    desc.text = f'Trace from {make} {model}'
    metatime = ET.SubElement(gpx, 'time')
    maxtime = datetime.fromtimestamp(0, timezone.utc)
    bounds = ET.SubElement(gpx, 'bounds')
    minlat, maxlat = 100, -100
    minlon, maxlon = 200, -200

    if generate_track:
        track = ET.SubElement(gpx, 'trk')
        container = ET.SubElement(track, 'trkseg')
        point = 'trkpt'
    else:
        container = gpx
        point = 'wpt'

    if isinstance(locdata, Mapping):
        iterator = locdata.items()
    else:
        iterator = enumerate(locdata)

    for i, fix in iterator:
        if fix['active'] not in b'AIX':
            continue

        lat, lon = fix['lat'], fix['lon']
        minlat, maxlat = min(minlat, lat), max(maxlat, lat)
        minlon, maxlon = min(minlon, lon), max(maxlon, lon)

        trkpt = ET.SubElement(container, point,
                              lat=f"{lat:.6f}", lon=f"{lon:.6f}")
        ts = fix['ts']
        maxtime = max(maxtime, ts)
        time = ET.SubElement(trkpt, 'time')
        time.text = ts.isoformat()
        course = ET.SubElement(trkpt, 'course')
        course.text = f"{fix['bearing']:.1f}"
        if photo := fix.get('photo'):
            comment = ET.SubElement(trkpt, 'name')
            comment.text = photo

        speed = ET.SubElement(trkpt, 'speed')
        speed.text = f"{fix['speed']:.1f}"

    bounds.set('minlat', f"{minlat:.6f}")
    bounds.set('maxlat', f"{maxlat:.6f}")
    bounds.set('minlon', f"{minlon:.6f}")
    bounds.set('maxlon', f"{maxlon:.6f}")
    metatime.text = maxtime.isoformat()
    return ET.ElementTree(gpx)


# Borked
def build_kml(locdata: dict, make: str, model: str):
    kml = ET.Element('kml', xmlns='http://earth.google.com/kml/2.2')
    doc = ET.SubElement(kml, 'Document')

    if isinstance(locdata, Mapping):
        iterator = locdata.items()
    else:
        iterator = enumerate(locdata)

    for i, fix in iterator:
        lat, lon = fix['lat'], fix['lon']
        icon_name = f'icon_{i}'

        photo = ET.SubElement(doc, 'Placemark')
        point = ET.SubElement(photo, 'Point')
        coordinates = ET.SubElement(point, 'coordinates')
        coordinates.text = f'{lon:.6f},{lat:.6f}'

        # vv = ET.SubElement(photo, 'ViewVolume')
        # ET.SubElement(vv, 'near').text = '1000'
        # ET.SubElement(vv, 'leftFov').text = '-60'
        # ET.SubElement(vv, 'rightFov').text = '60'
        # ET.SubElement(vv, 'bottomFov').text = '-45'
        # ET.SubElement(vv, 'topFov').text = '45'

        photofile = fix['photo']
        if os.path.exists(photofile):
            style = ET.SubElement(photo, 'Style', id=icon_name)
            iconstyle = ET.SubElement(style, 'IconStyle')
            icon = ET.SubElement(iconstyle, 'Icon')
            href = ET.SubElement(icon, 'href')
            href.text = os.path.basename(photofile)

            ET.SubElement(iconstyle, 'scale').text = '3.0'
            ET.SubElement(photo, 'styleUrl').text = '#'+icon_name
            ET.SubElement(photo, 'name').text = photofile

            # with open(photofile, 'rb') as fp:
            #     with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #         icon = ET.SubElement(photo, 'Icon')
            #         href = ET.SubElement(icon, 'href')
            #         href.text = f'data:image/jpeg;base64,'+base64.b64encode(mm).decode("us-ascii")

        # thumb = fix.get('thumbnail')
        # if thumb:
        #     enc = base64.b64encode(thumb).decode("us-ascii")
        #     desc = ET.SubElement(photo, 'description')
        #     desc.text = f'<img src="data:image/jpeg;base64,{enc}">'

    return ET.ElementTree(kml)


def extrapolate_locdata(data1: dict, data2: dict) -> dict:
    mx = data1["mx"]-(data2["mx"]-data1["mx"])
    my = data1["my"]-(data2["my"]-data1["my"])
    lon, lat = metric_lonlat(mx, my)
    deltabearing = (data1["bearing"]-data2["bearing"]) % 360
    if deltabearing > 180:
        deltabearing = deltabearing-360

    tsdiff = data1["ts"]-(data2["ts"]-data1["ts"])
    return dict(
        ts=tsdiff, posix_clock=tsdiff.timestamp(),
        mx=mx, my=my, lat=lat, lon=lon, active=b'X',
        bearing=(data1["bearing"]-deltabearing) % 360,
        speed=max(data1["speed"]-(data2["speed"]-data1["speed"]), 0),
        metric=0, prevdist=0)


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def interpolate_track(locdata: dict, res: float = 1.0, num: int = 0, deg: int = 1) -> dict:
    if len(locdata) <= deg:
        # Not enough data to interpolate
        return locdata

    # Convert track to gpx_interpolate format
    gpxdata = {'lat': [], 'lon': [], 'tstamp': [], 'ele': [],
               'tzinfo': timezone.utc}
    for point in locdata.values():
        gpxdata['lat'].append(point['lat'])
        gpxdata['lon'].append(point['lon'])
        gpxdata['tstamp'].append(point['posix_clock'])

    if all_equal(gpxdata['lat']) and all_equal(gpxdata['lon']):
        return locdata

    interpolated = gpx_interpolate(gpxdata, res, num, deg)
    speeds = gpx_calculate_speed(interpolated)

    # Convert back
    outdata = {}
    distance = totaldist = 0.0
    bearing = locdata[0]['bearing']

    for i, tstamp in enumerate(interpolated['tstamp']):
        lon, lat = interpolated['lon'][i], interpolated['lat'][i]
        mx, my = lonlat_metric(lon, lat)

        if i > 0:
            new_bearing, _, distance = WGS84.inv(interpolated['lon'][i-1],
                                                 interpolated['lat'][i-1],
                                                 lon, lat)
            if speeds[i] > 0.1:
                bearing = new_bearing % 360

        totaldist += distance
        outdata[i] = dict(ts=datetime.fromtimestamp(tstamp, timezone.utc),
                          posix_clock=tstamp, active=b'A',
                          lon=lon, lat=lat, mx=mx, my=my,
                          speed=speeds[i], bearing=bearing, prevdist=distance,
                          metric=totaldist)
    return outdata


def extract_gps(input_ts_file: os.PathLike, tzone, logger,
                make, model, device_override,
                length=0) -> gpxpy.gpx.GPXTrackSegment:
    '''Extract the GPS data from the specified input file.'''
    device, detected_make, detected_model, offsets = detect_file_type(
        input_ts_file, device_override, logger)
    logger.debug('Detected %s %s %s', detected_make, detected_model, device)
    make = make or detected_make
    model = model or detected_model

    if device == 'P':
        segment = get_gps_data_gopro(input_ts_file, device, tzone)
    elif device in "BVS":
        segment = get_gps_data_ts(input_ts_file, device, tzone, logger)
    elif device == "T":
        segment = get_gps_data_nt(input_ts_file, device, tzone,
                                  logger, offsets)
    elif device == "N":
        segment = get_gps_data_nmea(input_ts_file, device, tzone)
    elif device == "G":
        segment = get_gps_data_garmin(input_ts_file, device, tzone)
    else:
        segment = gpxpy.gpx.GPXTrackSegment()

    logger.debug("GPS data analysis ended; %d points", segment.get_points_no())
    # if length and packetno:
    #     logger.debug("Frames per point: %f", length/packetno)

    return segment


MERGE_EPSILON = timedelta(seconds=1)


def clean_gpx(gpxdata: gpxpy.gpx.GPX) -> gpxpy.gpx.GPX:
    """Remove garbage GPS data."""
    gpxlen = gpxdata.get_points_no()
    if gpxlen >= 2:
        last_trackno = last_segno = last_pos = -1
        last_point = None
        to_delete = []
        point: gpxpy.gpx.GPXTrackPoint
        for point, trackno, segno, pos in gpxdata.walk():
            if point.type_of_gpx_fix and point.type_of_gpx_fix == 'none':
                continue
            if last_point and last_trackno == trackno and last_segno == segno:
                dist = last_point.distance_2d(point)
                if dist > 10000:
                    logger.debug('lp: %s p: %s', last_point, point)
                    logger.debug('Dropping (bogus?) point %d %.1f meters away',
                                 pos, dist)
                    to_delete.append((trackno, segno, pos))
                    continue
            last_trackno, last_segno = trackno, segno
            last_point = point

        for trackno, segno, pos in reversed(to_delete):
            gpxdata.tracks[trackno].segments[segno].remove_point(pos)

        gpxdata.remove_empty()

    # Merge adjacent tracks if close in time
    prev_track = None
    for track in gpxdata.tracks:
        if prev_track:
            prev_start, prev_end = prev_track.get_time_bounds()
            this_start, this_end = track.get_time_bounds()

            if not this_start or not prev_end:
                continue

            print(prev_end, this_start)
            if prev_end <= this_start <= prev_end+MERGE_EPSILON:
                print('Merging', prev_track, track)
                for seg in track.segments:
                    prev_track.segments.append(seg)
                while len(prev_track.segments) > 1:
                    prev_track.join(0)
                track.segments = []
            else:
                prev_track = track
        else:
            prev_track = track

    gpxdata.remove_empty()
    return gpxdata


def clean_gps_data(locdata, length, fps, min_coverage, timeshift=0):
    '''Interpolate, extrapolate, and remove garbage GPS data.'''

    # Remove any insane coordinate values
    droplist = []
    for i, fix in locdata.items():
        if fix['active'] != b'A':
            droplist.append(i)

    for i in droplist:
        del locdata[i]

    fixcount = len(locdata)
    if fixcount >= 2:
        keylist = sorted(locdata)
        lats = [locdata[j]['lat'] for j in keylist]
        lons = [locdata[j]['lon'] for j in keylist]
        distances = WGS84.line_lengths(lats=lats, lons=lons)

        for i, prev_distance in enumerate(distances):
            # Probably garbage data if we've jumped > 10 km
            if prev_distance > 10000:
                logger.debug('Dropping (bogus?) point %d %.1f meters away', i,
                             prev_distance)
                del locdata[i+1]

    # Need at least 2 points to interpolate
    if len(locdata) < 2:
        logger.info(
            'Not enough GPS data for interpolation; need >= 2 points, got %d',
            len(locdata))
    elif len(locdata) < min_coverage*length*0.01/fps:
        logger.info(
            "Not enough GPS data for interpolation; %d%% needed, %f%% found",
            min_coverage, 100*len(locdata)/length*fps)

    locdata = interpolate_track(locdata)

    if len(locdata) >= 2:
        time_per_fix = (locdata[max(locdata)]['posix_clock'] -
                        locdata[min(locdata)]['posix_clock'])/len(locdata)
        extend = int(math.ceil(abs(timeshift)/time_per_fix))

        for i in range(min(locdata), min(locdata)-extend, -1):
            if i not in locdata:
                locdata[i] = extrapolate_locdata(locdata[i+1], locdata[i+2])
        for i in range(max(locdata)+1, int(1.1 * length / fps)):
            if i not in locdata:
                locdata[i] = extrapolate_locdata(locdata[i-1], locdata[i-2])

        keylist = sorted(locdata)
        lats = [locdata[j]['lat'] for j in keylist]
        lons = [locdata[j]['lon'] for j in keylist]
        distances = WGS84.line_lengths(lats=lats, lons=lons)

        for i, prev_distance in enumerate(distances):
            locdata[keylist[i+1]]['prevdist'] = prev_distance

        for i, cum_distance in enumerate(accumulate(distances)):
            locdata[keylist[i+1]]['metric'] = cum_distance

    return locdata


# Code so I can easily switch video and image processing backends
class VideoWrapper:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class OpenCVVideoWrapper(VideoWrapper):
    def __init__(self, path: os.PathLike, maskfile: os.PathLike = '',
                 rotate: float = 0, keep_aspect_ratio: bool = False,
                 crop_top: int = 0, crop_left: int = 0,
                 crop_bottom: int = 0, crop_right: int = 0):
        self.video = cv2.VideoCapture(str(path))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.mask = cv2.imread(maskfile, 0) if maskfile else None
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rotate = rotate
        self.keep_aspect_ratio = keep_aspect_ratio
        self.crop_top, self.crop_bottom = crop_top, crop_bottom
        self.crop_left, self.crop_right = crop_left, crop_right

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def __iter__(self):
        success, framedata = self.video.read()
        while success:
            if self.mask:
                cv2.bitwise_and(framedata, framedata, mask=self.mask)

            if self.rotate:
                orig_height, orig_width = framedata.shape[:2]
                framedata = imutils.rotate_bound(framedata, -self.rotate)
                height, width = framedata.shape[:2]
                nw, nh = rotatedRectWithMaxArea(
                    orig_width, orig_height, math.radians(-self.rotate))

                if self.keep_aspect_ratio:
                    orig_ratio = orig_width / orig_height
                    new_ratio = nw / nh
                    if new_ratio > orig_ratio:
                        nw = nh*orig_ratio
                    else:
                        nh = nw/orig_ratio

                x1 = int((width - nw)/2)
                y1 = int((height - nh)/2)

                x2 = int(x1 + nw)+1
                y2 = int(y1 + nh)+1

                framedata = framedata[y1:y2, x1:x2]

            if self.crop_left or self.crop_right or \
                 self.crop_bottom or self.crop_top:
                h, w = framedata.shape[:2]
                framedata = framedata[self.crop_top:h-self.crop_bottom,
                                      self.crop_left:w-self.crop_right]

            yield Image.fromarray(
                cv2.cvtColor(framedata, cv2.COLOR_BGR2RGB))
            success, framedata = self.video.read()

    def seek_frame(self, frame):
        self.video.set(1, frame)

    def close(self):
        self.video.release()


class AVVideoWrapper(VideoWrapper):
    def __init__(self, path: os.PathLike, maskfile: os.PathLike = '',
                 rotate: float = 0, keep_aspect_ratio: bool = False,
                 crop_top: int = 0, crop_left: int = 0,
                 crop_bottom: int = 0, crop_right: int = 0):
        self.container = av.open(str(path))
        self.vidstream: av.video.VideoStream = self.container.streams.video[0]
        self.vidstream.thread_type = 'AUTO'
        self.fps = self.vidstream.average_rate  # 1 / self.vidstream.time_base
        self.length = self.vidstream.frames
        self.wanted_time = 0
        self.current_time = 0
        self.mask = Image.open(maskfile) if maskfile else None

        stream = self.container.streams.video[0]
        self.width, self.height = stream.width, stream.height

        # Set up image filter graph
        self.graph = av.filter.Graph()
        tail = self.graph.add_buffer(template=self.vidstream)
        # range_filter = self.graph.add('colorspace', 'bt601-6-525:range=pc')
        # tail.link_to(range_filter)
        # tail = range_filter
        if rotate:
            angle = f'{-rotate}*PI/180'
            rotate_filter = self.graph.add(
                'rotate', f'{angle}:ow=rotw({angle}):oh=roth({angle})')
            tail.link_to(rotate_filter)
            tail = rotate_filter
            nw, nh = rotatedRectWithMaxArea(
                self.vidstream.width, self.vidstream.height,
                math.radians(-rotate))
            if keep_aspect_ratio:
                orig_ratio = self.vidstream.width / self.vidstream.height
                new_ratio = nw / nh
                if new_ratio > orig_ratio:
                    nw = nh*orig_ratio
                else:
                    nh = nw/orig_ratio

            # ffmpeg's default is to center the crop, which is what we want
            crop_filter = self.graph.add('crop', f'w={int(nw)}:h={int(nh)}')
            tail.link_to(crop_filter)
            tail = crop_filter

        if crop_left or crop_right or crop_top or crop_bottom:
            img_crop_filter = self.graph.add(
                'crop', f'w=iw-{int(crop_right+crop_left)}:'
                f'h=ih-{int(crop_bottom+crop_top)}:'
                f'x={crop_left}:y={crop_top}')
            tail.link_to(img_crop_filter)
            tail = img_crop_filter

        sink = self.graph.add('buffersink')
        tail.link_to(sink)
        self.graph.configure()

    def __del__(self):
        self.close()

    def __iter__(self):
        # May need to move forward from keyframe to get the right frame
        frame: av.VideoFrame
        for frame in self.container.decode(video=0):
            self.current_time = frame.pts * frame.time_base
            # print('At', self.current_time, 'want', self.wanted_time)
            if self.current_time >= self.wanted_time and (
                    self.wanted_time < (self.current_time + frame.time_base)):
                # frame = frame.reformat(format='yuv420p')
                if self.mask:
                    img = frame.to_image()
                    img = Image.composite(img, img, self.mask)
                    frame = av.video.frame.VideoFrame.from_image(img)

                self.graph.push(frame)
                filtered_frame = self.graph.pull()
                yield filtered_frame.to_image()

    def seek_frame(self, frameno):
        self.wanted_time = frameno / self.fps
        # Probably don't want to seek if wanted frame is in near future
        if self.wanted_time <= self.current_time or (
                self.wanted_time > self.current_time + 2):
            seekpoint = int(self.wanted_time / self.vidstream.time_base)
            self.container.seek(seekpoint, stream=self.vidstream)

    def close(self):
        if self.container:
            self.container.close()
            self.container = None


FNPARSER = re.compile('(\d{4})_?(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})')


def guess_start_time(path: os.PathLike, first_fix_time: datetime=None,
                     tz: float = 0) -> datetime:
    path = Path(path)
    tzinfo = timezone(timedelta(hours=tz))
    if match := FNPARSER.match(path.stem):
        year, month, day, h, m, s = (int(x) for x in match.groups())
        ts = datetime(year, month, day, h, m, s, 0, tzinfo)
        if first_fix_time and first_fix_time != ts:
            logger.info('guessed %s log starts at %s', ts, first_fix_time)
        return ts
    stinfo = path.stat()
    ts = datetime.fromtimestamp(stinfo.st_ctime, tzinfo)
    if first_fix_time and first_fix_time != ts:
        logger.info('guessed %s log starts at %s', ts, first_fix_time)
    return ts


def galleryview_path(frame: int):
    return f'frame{frame:06d}.html'


def create_galleryview(video: VideoWrapper, frame: int,
                       posinfo: gpxpy.gpx.GPXTrackPoint,
                       basedir: os.PathLike, prevframe: int,
                       nextframe: int, lastframe: int):
    import folium

    video.seek_frame(frame)
    vidframe = next(iter(video))
    imgpath = f'img{frame:06d}.jpg'
    vidframe.save(Path(basedir) / imgpath)

    coords = (posinfo.latitude, posinfo.longitude)
    folium_map = folium.Map(location=coords, zoom_start=18)
    folium.Marker(coords, popup=f'frame {frame}').add_to(folium_map)
    map_html = folium_map._repr_html_()

    navbar = ''
    if prevframe is not None:
        navbar += f'<a href="{galleryview_path(0)}">&lt;&lt;</a> <a href="{galleryview_path(prevframe)}">&lt;</a> '
    else:
        navbar += f'<a href="{galleryview_path(0)}">&lt;&lt;</a> &lt; '

    if nextframe is not None:
        navbar += f'<a href="{galleryview_path(nextframe)}">&gt;</a> <a href="{galleryview_path(lastframe)}">&gt;&gt;</a> '
    else:
        navbar += f'&gt; <a href="{galleryview_path(lastframe)}">&gt;&gt;</a> '

    navbar += f'{basedir} Frame: {frame:06d} {coords[0]:.6f} {coords[1]:.6f} Timestamp: {posinfo["ts"]} '

    pct = 60
    livepage = f'<html><body><div>{navbar}</div><div style="width: {pct}%; float:left;"><img src="{imgpath}" style="width: 100%; height: auto"></div><div style="width: {100-pct}%;float: right;">{map_html}</div></body></html>'
    fname = galleryview_path(frame)
    with open(Path(basedir) / fname, 'w') as fp:
        fp.write(livepage)
    return fname


def locdata_to_gpxsegment(locdata) -> gpxpy.gpx.GPXTrackSegment:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    for entry in locdata.values():
        # logger.debug('entry: %s', entry)
        gpx_point = gpxpy.gpx.GPXTrackPoint(entry.get('lat'),
                                            entry.get('lon'),
                                            elevation=entry.get('ele'),
                                            time=entry['ts'],
                                            speed=entry.get('speed'))
        gpx_point.course = entry.get('bearing')
        gpx_point.type_of_gpx_fix = entry.get('fix')
        # logger.debug('  point: %s', repr(gpx_point))
        gpx_segment.points.append(gpx_point)

    return gpx_segment


def fixup_gpx_order(gpx: gpxpy.gpx.GPX):
    """Ensure tracks are in chronological order in tracklist."""
    if not gpx or not gpx.tracks:
        return gpx

    for track in gpx.tracks:
        track.segments.sort(key=gpxpy.gpx.GPXTrackSegment.get_time_bounds)

    gpx.tracks.sort(key=gpxpy.gpx.GPXTrack.get_time_bounds)
    return gpx


def get_gps_data(input_ts_file: os.PathLike, tzone: timezone,
                 make, model, device_override) -> gpxpy.gpx.GPXTrackSegment:
    logger.debug('Getting GPS data from %s', input_ts_file)
    segment = extract_gps(input_ts_file, tzone, logger,
                          make, model, device_override)
    return segment


def get_all_gps(inputfiles: Iterable[os.PathLike], parallel: int, make: str,
                model: str, device_override: str,
                tz: float) -> (gpxpy.gpx.GPX, dict):
    tzone = timezone(timedelta(hours=tz))
    ldmap = {}

    if parallel == 1:
        segments = {input_ts_file: get_gps_data(input_ts_file, tzone, make,
                                                model, device_override)
                    for input_ts_file in inputfiles}
    else:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(get_gps_data, input_ts_file,
                                       tzone, make, model, device_override):
                       input_ts_file for input_ts_file in inputfiles}
            segments = {futures[future]: future.result() for future
                        in concurrent.futures.as_completed(futures)}

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    for input_ts_file, gpx_segment in segments.items():
        if gpx_segment and gpx_segment.get_points_no():
            start_time = gpx_segment.points[0].time
            end_time = gpx_segment.points[-1].time
            ldmap[input_ts_file] = (start_time, end_time)
            gpx_track.segments.append(gpx_segment)

    gpx = fixup_gpx_order(gpx)
    gpx = clean_gpx(gpx)
    return gpx, ldmap


INTERPOLATE_EPSILON = timedelta(seconds=120)
SCAN_EPSILON = timedelta(seconds=5)


def interpolate_location(gpxdata: Union[gpxpy.gpx.GPX, gpxpy.gpx.GPXTrack],
                         timestamp: datetime) -> gpxpy.gpx.GPXTrackPoint:
    """Get interpolated location from `gpxdata` at `timestamp`."""
    point: gpxpy.gpx.GPXTrackPoint
    prev_point: gpxpy.gpx.GPXTrackPoint = None
    tracks_to_scan: List[gpxpy.gpx.GPXTrack] = None

    if hasattr(gpxdata, 'tracks'):
        for track in gpxdata.tracks:
            start, end = track.get_time_bounds()
            if (start - SCAN_EPSILON) <= timestamp <= (end + SCAN_EPSILON):
                tracks_to_scan = [track]
                break

        if not tracks_to_scan:
            tracks_to_scan = gpxdata.tracks
    else:
        tracks_to_scan = [gpxdata]

    # XXX - probably faster to do some sort of bisection search
    for track in tracks_to_scan:
        for point in track.walk(only_points=True):
            if point.type_of_gpx_fix and point.type_of_gpx_fix == 'none':
                # Skip points where we don't have a fix
                continue
            if point.time == timestamp:
                return point
            elif timestamp > point.time >= (timestamp - INTERPOLATE_EPSILON):
                # logger.debug('setting prev_point %s for %s', point, timestamp)
                prev_point = point
                # print(prev_point)
            elif prev_point and point.time >= timestamp:
                # Linear interpolation - probably should account for speed change?
                speed = point.speed_between(prev_point)
                course = prev_point.course_between(point)
                numerator = (timestamp - prev_point.time).total_seconds()
                denominator = point.time_difference(prev_point)
                prop = numerator/denominator

                x1, y1 = lonlat_metric(prev_point.longitude, prev_point.latitude)
                x2, y2 = lonlat_metric(point.longitude, point.latitude)
                deltax, deltay = x2-x1, y2-y1
                lon, lat = metric_lonlat(x1+deltax*prop, y1+deltay*prop)

                newpoint = gpxpy.gpx.GPXTrackPoint(lat, lon, time=timestamp,
                                                   speed=speed)
                newpoint.course = course

                if point.elevation is not None and \
                   prev_point.elevation is not None:
                    deltaz = point.elevation - prev_point.elevation
                    newpoint.elevation = prev_point.elevation+deltaz*prop
                    newpoint.type_of_gpx_fix = '3d'
                else:
                    newpoint.type_of_gpx_fix = '2d'

                # Override fix type if both points agree
                if point.type_of_gpx_fix == prev_point.type_of_gpx_fix:
                    newpoint.type_of_gpx_fix = point.type_of_gpx_fix

                if point.satellites and prev_point.satellites:
                    newpoint.satellites = min(point.satellites,
                                              prev_point.satellites)

                if point.horizontal_dilution and prev_point.horizontal_dilution:
                    newpoint.horizontal_dilution = math.sqrt(
                        point.horizontal_dilution**2 +
                        prev_point.horizontal_dilution**2)

                if point.vertical_dilution and prev_point.vertical_dilution:
                    newpoint.vertical_dilution = math.sqrt(
                        point.vertical_dilution**2 +
                        prev_point.vertical_dilution**2)

                if point.position_dilution and prev_point.position_dilution:
                    newpoint.position_dilution = math.sqrt(
                        point.position_dilution**2 +
                        prev_point.position_dilution**2)

                # print(newpoint)
                return newpoint

    if not prev_point:
        # If we get here, no points had a valid fix
        return None

    # If we get here we need to extrapolate
    elapsed_time = (timestamp - prev_point.time).total_seconds()
    distance = prev_point.speed*elapsed_time
    newlon, newlat, backaz = WGS84.fwd(prev_point.longitude,
                                       prev_point.latitude, prev_point.course,
                                       distance)

    newpoint = gpxpy.gpx.GPXTrackPoint(newlat, newlon, speed=prev_point.speed,
                                       time=timestamp)
    newpoint.course = prev_point.course
    newpoint.type_of_gpx_fix = prev_point.type_of_gpx_fix
    if prev_point.elevation is not None:
        newpoint.elevation = prev_point.elevation
    # logger.debug('Extrapolated from %s %1.fm %1.f° to %s', prev_point,
    #              distance, prev_point.course, newpoint)
    return newpoint


def process_video(input_ts_file: str, folder: str, thumbnails: bool=False,
                  maskfile: str = '', make='', model='', kml_out=False,
                  device_override='', tz: float=0,
                  crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
                  sampling_interval: float=0.5, min_points=5,
                  min_speed=-1, timeshift: float=0, metric_distance: float=0,
                  turning_angle: float=0, suppress_cv2_warnings=False,
                  use_sampling_interval=True, config=None, sensor_width=None,
                  bearing_modifier: float=0, use_speed=False, limit: int=0,
                  max_aperture=None, rotate: float=0, output_format='jpeg',
                  focal_length=None, min_coverage=90,
                  keep_aspect_ratio=False, gallery=False,
                  keep_night_photos=False,
                  geofence_spec: geofence.Geofence = None,
                  # csv_out=False,
                  gpx_out=False, dry_run=False,
                  external_gps_data: gpxpy.gpx.GPX = None,
                  internal_gps_data: gpxpy.gpx.GPX = None,
                  logger=logger, ldmap=None) -> int:
    # logger = multiprocessing.get_logger()
    # logger = logging.getLogger('process_video.'+input_ts_file)
    logger.info('Processing %s', input_ts_file)

    shiftdelta = timedelta(seconds=timeshift)
    vidcontext = (suppress_stdout_stderr if suppress_cv2_warnings else
                  contextlib.nullcontext)
    exifdata = {}

    with vidcontext():
        try:
            video = AVVideoWrapper(input_ts_file, maskfile, rotate,
                                   keep_aspect_ratio,
                                   crop_top, crop_left,
                                   crop_bottom, crop_right)
            video_iterator = iter(video)
        except av.error.InvalidDataError as exc:
            logger.warning('%s is invalid: %s', input_ts_file, exc)
            return 0
        except av.error.FFmpegError as exc:
            logger.warning('%s error: %s', input_ts_file, exc)
            return 0

    fps, length = video.fps, video.length
    logger.debug('FPS: %d; LEN: %d', fps, length)

    if not video.length:
        logger.warning('%s is empty', input_ts_file)
        return 0

    # XXX - move this code so we don't have to execute it each time
    if not config:
        config = configparser.ConfigParser()

    modstr = f'{make} {model}'
    max_aperture = config.get(modstr, 'max_aperture',
                               fallback=max_aperture)
    if max_aperture:
        # Convert to APEX - # of stops from f/1
        apex_max_aperture = math.log(max_aperture**2, 2)
        # F numbers are rounded to 2 SF, so reverse the rounding
        # and store precisely
        apex_max_aperture = Fraction(round(apex_max_aperture)*6, 6)
        exifdata['max aperture'] = apex_max_aperture

    sensor_width = config.get(modstr, 'sensor_width',
                              fallback=sensor_width)
    if sensor_width:
        width, height = video.width, video.height
        pixel_width = sensor_width/width/10
        exifdata['focal plane x resolution'] = pixel_width
        exifdata['focal plane y resolution'] = pixel_width

    focal_length = config.get(modstr, 'focal_length', fallback=focal_length)
    if focal_length:
        exifdata['focal length'] = focal_length

    if sensor_width and focal_length:
        crop_factor = 35/sensor_width
        exifdata['35mm equivalent focal length'] = focal_length*crop_factor

    # XXX end of code to move

    interval = int(sampling_interval*fps)
    if interval == 0:
        interval = 1

    # tzone = timezone(timedelta(hours=tz))
    if not internal_gps_data:
        logger.warning('No GPS data found in %s', input_ts_file)
        return 0

    length_seconds = float(length/fps)
    filename_based_start = guess_start_time(input_ts_file, tz=tz)
    first_time, last_time = ldmap.get(input_ts_file, (None, None))

    if first_time and last_time:
        if first_time > last_time:
            logger.warning('GPS ends before it began? %s–%s',
                           first_time, last_time)

        if (last_time-first_time).total_seconds() > length/fps:
            logger.warning('Times exceed video length: %s–%s',
                           first_time, last_time)

        if first_time and \
           abs(filename_based_start - first_time).total_seconds() > 600:
            # Embedded start time is way off
            logger.warning('Filename says starts at %s, track starts at %s',
                           filename_based_start, first_time)

    if not first_time and not last_time:
        start_time = first_time = filename_based_start
        last_time = filename_based_start + timedelta(seconds=length_seconds)
        logger.debug('Guessed video range %s–%s', start_time, last_time)
    else:
        start_time = min(first_time,
                         last_time-timedelta(seconds=length_seconds-1))
        if start_time != filename_based_start:
            logger.debug('start from onboard GPS: %s  guessed video start: %s',
                         start_time, filename_based_start)

    ### Logging

    if not os.path.exists(folder) and not dry_run:
        os.makedirs(folder)

    vidext = EXTENSIONS.get(output_format.lower(), output_format.lower())
    fnbase = Path(folder) / (Path(input_ts_file).stem + '_')

    if gallery:
        gallerydir = Path(folder) / ('gal-'+(Path(input_ts_file).stem))
        if not os.path.exists(gallerydir):
            os.makedirs(gallerydir)

    position_data = internal_gps_data
    if external_gps_data:
        end_time = start_time + timedelta(seconds=float((length-1)/fps))
        shift_start, shift_end = start_time+shiftdelta, end_time+shiftdelta
        logger.debug('shifted video times: %s–%s', shift_start, shift_end)
        for track in external_gps_data.tracks:
            ext_start, ext_end = track.get_time_bounds()
            logger.debug(' %s times: %s–%s', track, ext_start, ext_end)
            if ext_start <= shift_start and ext_end >= shift_end:
                logger.debug('using external track %s', track)
                position_data = track
                break

        if position_data == internal_gps_data:
            ext_start, ext_end = external_gps_data.get_time_bounds()
            logger.warning('Ignoring external GPX data for track %s [%s–%s]: '
                           'out of time bounds [%s–%s].', input_ts_file,
                           shift_start, shift_end, ext_start, ext_end)

    if position_data.get_points_no() < min_points:
        logger.warning("Not enough GPS data for frame extraction: %s",
                       input_ts_file)
        return 0

    if dry_run:
        logger.info("Would be extracting from %s now", input_ts_file)
    else:
        logger.info("Extraction started: %s", input_ts_file)

    framecount = 0
    count = 0
    turning = False
    useframe = False
    raw_thumbnail = None

    logger.debug('%s: guessed first frame at %s', input_ts_file, start_time)

    photos = []
    success = True
    browser = False
    last_position = None

    while success and framecount < length and (not limit or count < limit):
        # Interpolate time and coordinates
        current_time = start_time + timedelta(seconds=float(framecount/fps))
        shifted_time = current_time + shiftdelta

        position = interpolate_location(position_data, shifted_time)
        if not position:
            logger.debug('no fix for frame %d at %s', framecount, shifted_time)
            framecount += int(1 * fps)
            continue

        if not keep_night_photos and geofence.image_is_dark(
                position.latitude, position.longitude, shifted_time):
            logger.debug('skipping frame %d: dark at %s',
                         framecount, position.time)
            # We can skip ahead a bit
            framecount += int(15 * fps)
            continue

        if geofence_spec and geofence_spec.position_in_fence(
                position.latitude, position.longitude):
            logger.debug('skipping frame %d: fenced', framecount)
            # We can skip ahead a bit
            framecount += int(1 * fps)
            continue

        distance = position.distance_2d(last_position) if last_position else 0
        if framecount % fps == 0:
            logger.debug('fr %d ts %s target %.2fm at %.2fm',
                         framecount, shifted_time,
                         metric_distance, distance)
        # logger.debug('%.3f %.3f', next_distance, posinfo['metric'])
        # logger.debug(currentpos)
        # logger.debug(lastframe)
        # logger.debug(posinfo)

        if gallery:
            fpath = create_galleryview(
                video, framecount, position, gallerydir,
                framecount-1 if framecount > 0 else None,
                framecount+1 if framecount < length-1 else None,
                length-1)
            if not browser:
                webbrowser.open(str(fpath))
                browser = True

        if use_sampling_interval:
            if not last_position or (
                    (shifted_time - last_position.time).total_seconds() >
                    sampling_interval):
                useframe = True

        if metric_distance and not turning_angle:
            if last_position and distance < metric_distance:
                framecount += 1
                continue
            else:
                useframe = True
        elif turning_angle:
            # Peek ahead
            # anglediff = (posinfo['bearing'] - locnext['bearing']) % 360
            # if not turning and turning_angle < min(anglediff, 360-anglediff):
            #     useframe = True
            #     turning = True
            if last_position:
                # Not sure how travel_distance was different?
                speed = position.speed
                anglediff = (last_position.course - position.course) % 360
                if distance < 1 or speed < 0.5:
                    framecount += 1
                    continue
                elif turning_angle < min(anglediff, 360-anglediff):
                    # Check for near 180 angles here?
                    logger.debug('turning')
                    useframe = True
                    turning = True
                elif (metric_distance and distance >= metric_distance):
                    useframe = True
                    turning = False
                # elif turning: # Capture end of turn
                #     useframe = True
                #     turning = False
                else:
                    framecount += 1
                    continue

                logger.debug('%d angle: %.3f dist: %.3f v: %.1f',
                             framecount, min(anglediff, 360-anglediff),
                             distance, speed)
            else:
                useframe = True

        if useframe:
            with vidcontext():
                video.seek_frame(framecount)
                pil_image = next(video_iterator)
                success = bool(pil_image)
            if not success:
                break

            jpgname = f'{fnbase}{count:06d}.{vidext}'
            logger.debug('Would save fr %d: %s' if dry_run
                         else 'Saving fr %d: %s', framecount, jpgname)

            if thumbnails and output_format == 'jpeg':
                thumbnail = pil_image.copy()
                thumbnail.thumbnail((THUMB_SIZE, THUMB_SIZE))
                o = io.BytesIO()
                thumbnail.save(o, 'jpeg', quality=THUMB_QUALITY)
                raw_thumbnail = o.getvalue()
                # posinfo['thumbnail'] = raw_thumbnail

            width, height = pil_image.size
            exif_bytes = build_exif_data(
                position, make, model, width, height,
                exifdata, raw_thumbnail, bearing_modifier)

            if not dry_run:
                pil_image.save(jpgname, output_format, **PIL_SAVE_SETTINGS,
                               exif=exif_bytes)
                ftime = position.time.timestamp()
                os.utime(jpgname, (ftime, ftime))

            count += 1
            photos.append((position, jpgname))
            last_position = position

        # End of loop
        if not metric_distance and not turning_angle:
            framecount += int(sampling_interval*fps)
            useframe = True
        else:
            framecount += 1
            useframe = False

    video.close()

    if dry_run:
        logger.info('%s processed; would have extracted %d image(s)',
                    input_ts_file, count)
        return True

    logger.info('%s processed; %d image(s) extracted', input_ts_file, count)
    return count


def process_video_with_exceptions(infile: os.PathLike, folder: os.PathLike,
                                  *args, **kwargs) -> int:
    try:
        return process_video(infile, folder, *args, **kwargs)
    except Exception:
        tb = cgitb.text(sys.exc_info())
        print(tb, file=sys.stderr)
        raise


def find_files(filelist: Iterable[os.PathLike],
               recursive=False) -> List[os.PathLike]:
    inputfiles = []
    for fname in filelist:
        if '*' in fname or '?' in fname:
            if not recursive:
                # If they used '**', allow it even if not args.recursive
                inputfiles += glob.iglob(fname, recursive=True)
                continue

            dirname, filepart = os.path.split(fname)
            # If final part of the filename is a glob, glob the whole shebang
            if '*' in filepart or '?' in filepart:
                dirname, filepart = fname, ''

            for d in glob.iglob(dirname):
                path = Path(d) / filepart
                if path.is_file():
                    inputfiles.append(path)
                elif path.is_dir() and not path.is_link():
                    # Shell glob would have acted as below if --recursive on
                    for root, dirs, files in os.walk(path):
                        rootpath = Path(root)
                        inputfiles += (
                            rootpath / f for f in files
                            if VALID_FILENAME.match(f))
                elif path.is_link():
                    logger.warning('not recursing into symlink %s', path)
                else:
                    logger.error('%s unsuitable, aborting', path)
                    sys.exit(1)

        # Must be a normal filename
        p = Path(fname)
        if p.is_file():
            inputfiles.append(p)
        elif p.is_dir():
            if recursive:
                for root, dirs, files in os.walk(p):
                    rootpath = Path(root)
                    inputfiles += (
                        rootpath / f for f in files
                        if VALID_FILENAME.match(str(f)))
            else:
                inputfiles += (f for f in p.iterdir()
                               if VALID_FILENAME.match(f.stem + f.suffix))
        else:
            logger.error("Can't find input file: %s", fname)
            sys.exit(1)

    inputfiles.sort()
    return inputfiles


def get_opener(filename: str) -> Callable[..., TextIO]:
    """Get appropriate open function for `filename` based on type."""
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        return gzip.open
    elif ext == '.bz2':
        return bz2.open
    elif ext == '.lzma':
        return lzma.open
    else:
        return open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', type=str,
                        help='input file or folder(s)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='descend into subdirectories too')
    parser.add_argument('--sampling-interval', '--sampling_interval',
                        default=0.5, type=float,
                        help='distance between images in seconds')
    parser.add_argument('--folder', '--output', default='output', type=str,
                        help='output folder, will be created if missing; default is "output"')
    parser.add_argument('--output-format', default='jpeg', type=str,
                        help='output format to use; jpeg is default')
    parser.add_argument('--timeshift', default=0, type=float,
                        help='time shift in seconds, if the GPS and video seem out of sync')
    parser.add_argument('--timezone', default=0, type=Decimal, dest='tz',
                        help='timezone difference in hours. Depends on video source, some provide GMT, others local')
    parser.add_argument('--min-speed', '--min_speed', default=-1, type=float,
                        help='minimum speed in m/s to filter out stops')
    parser.add_argument('--bearing-modifier', '--bearing_modifier',
                        default=0, type=float,
                        help='set to 180 if rear camera')
    parser.add_argument('--min-coverage', '--min_coverage',
                        default=90, type=int,
                        help='percentage - how much video must have GPS data in order to interpolate missing')
    parser.add_argument('--min-points', '--min_points', default=5, type=int,
                        help='how many points to allow video extraction')
    parser.add_argument('--metric-distance', '--metric_distance', default=0,
                        type=int, help='distance between images, overrides sampling_interval')
    # parser.add_argument('--csv', action='store_true', dest='csv_out',
    #                     help="create csv from coordinates before and after interpolation.")
    parser.add_argument('--gpx', action='store_true', dest='gpx_out',
                        help="create GPX from coordinates before and after interpolation.")
    parser.add_argument('--suppress-cv2-warnings', '--suppress_cv2_warnings',
                        action='store_true', help="If disabled, will show lot of harmless warnings in console. Known to cause issues on Windows.")
    parser.add_argument('--device-override', '--device_override', default='',
                        type=str, help='force treatment as specific device, B for B4k, V for Viofo')
    parser.add_argument('--mask', type=str, dest='maskfile',
                        help='masking image, must be same dimensionally as video')
    parser.add_argument('--crop-left', '--crop_left', default=0, type=int,
                        help='number of pixels to crop from left')
    parser.add_argument('--crop-right', '--crop_right', default=0, type=int,
                        help='number of pixels to crop from right')
    parser.add_argument('--crop-top', '--crop_top', default=0, type=int,
                        help='number of pixels to crop from top')
    parser.add_argument('--crop-bottom', '--crop_bottom', default=0, type=int,
                        help='number of pixels to crop from bottom')
    parser.add_argument('--rotate', default=0, type=float,
                        help='rotate images; negative to rotate clockwise')
    parser.add_argument('--keep-aspect-ratio', action='store_true',
                        help='for rotated images, increase crop to maintain original aspect ratio')
    parser.add_argument('--limit', default=0, type=int,
                        help='limit number of images to extract per video; useful for testing settings')
    parser.add_argument('--make', type=str,
                        help='set camera make to be written in EXIF')
    parser.add_argument('--model', type=str,
                        help='set camera model to be written in EXIF')
    parser.add_argument('--turning-angle', '--turning_angle',
                        default=0, type=int,
                        help="override metric_distance when bearing changes by specified number of degrees")
    parser.add_argument('--parallel', '-j', default=1, action='store',
                        type=int, metavar='PROCESSES', nargs='?',
                        const=None,
                        help='process in parallel; if PROCESSES omitted, use as many processes as there are CPU cores/threads')
    parser.add_argument('--use-speed', action='store_true',
                        help='include recorded speed in the output images')
    parser.add_argument('--max-aperture', action='store', type=Decimal,
                        help='camera maximum aperture (F number)')
    parser.add_argument('--sensor-width', action='store', type=Decimal,
                        help='camera sensor width (mm)')
    parser.add_argument('--focal-length', action='store', type=Decimal,
                        help='camera lens focal length (mm, not adjusted)')
    parser.add_argument('--configfile', action='store', type=str,
                        help='use the specified configuration file')
    parser.add_argument('--thumbnails', action='store_true',
                        help='store thumbnails too')
    parser.add_argument('--kml', action='store_true', dest='kml_out',
                        help='produce a KML file showing photo locations')
    parser.add_argument('--fence', action='store', type=Path,
                        help='file to read geofences from')
    parser.add_argument('--keep-night-photos', action='store_true',
                        help="don't filter out photos taken before "
                        "local sunrise/after local sunset")
    parser.add_argument('--external-gpx',
                        help='external GPX file to use for position fixes')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quiet', '-q', action='store_const', dest='loglevel',
                       const=logging.WARNING, default=logging.INFO,
                       help='only show warnings and errors')
    group.add_argument('--debug', '-v', '--verbose',
                       action='store_const', dest='loglevel',
                       const=logging.DEBUG, default=logging.INFO,
                       help='show debugging messages too')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help="run but don't save anything to disk")
    parser.add_argument('--gallery', action='store_true',
                        help='create a gallery of all frames - useful for identifying correct time offset')

    args = parser.parse_args()
    # print(args)

    # logger = multiprocessing.log_to_stderr()
    logger.setLevel(args.loglevel)

    inputfiles = find_files(args.input, recursive=args.recursive)

    if not inputfiles:
        logger.error('No files to process.')
        sys.exit(1)

    # print(inputfiles, len(inputfiles))
    logger.info('Found %d file(s) to process.', len(inputfiles))

    configfiles = [args.configfile] if args.configfile else []

    if not configfiles:
        xdgconfdir = os.getenv('XDG_CONFIG_HOME', '~/.config')
        xdgconfdirs = os.getenv('XDG_CONFIG_DIRS', '/etc/xdg').split(':')

        configfiles = [Path(x, 'dashcam-photos.conf') for x in xdgconfdirs]
        configfiles.append(Path(xdgconfdir, 'dashcam-photos.conf'))

    config = configparser.ConfigParser()
    config.read(configfiles)

    modstr = f'{args.make} {args.model}'
    make = config.get(modstr, 'make', fallback=args.make)
    model = config.get(modstr, 'model', fallback=args.model)

    use_sampling_interval = (not args.metric_distance)

    if args.dry_run:
        logger.warning('*** Dry run - no data saved ***')

    if args.fence:
        if not args.fence.is_file():
            print(args.fence, 'not found', file=sys.stderr)
            sys.exit(1)

        geofence_spec = geofence.parse_geofence_spec(args.fence.read_text(),
                                                     args.fence)
        if not geofence_spec:
            logger.error('No fence information found in %s, aborting.',
                         args.fence)
            sys.exit(1)
    else:
        geofence_spec = None

    if args.external_gpx:
        logger.info('Reading external GPX file %s', args.external_gpx)
        gpx_opener = get_opener(args.external_gpx)
        with gpx_opener(args.external_gpx, 'rt') as gpxfile:
            external_gps_data = gpxpy.parse(gpxfile)
            if external_gps_data:
                external_gps_data = fixup_gpx_order(external_gps_data)
    else:
        external_gps_data = None

    logger.info('Reading GPS data from video files')
    internal_gps_data, ldmap = get_all_gps(
        inputfiles, args.parallel, make, model,
        args.device_override, args.tz)

    if args.gpx_out and not args.dry_run:
        if make and model:
            internal_gps_data.name = f'GPS data extracted from {make} {model}'
        gpxtree = internal_gps_data.to_xml(version='1.0')
        if not os.path.exists(args.folder):
            os.makedirs(args.folder)

        outfile = os.path.join(args.folder, 'extracted_tracks.gpx')
        logger.debug('Saving extracted location data to %s', outfile)
        with open(outfile, 'w') as fob:
            fob.write(gpxtree)

    kwargs = {'config': config, 'geofence_spec': geofence_spec,
              'use_sampling_interval': use_sampling_interval,
              'make': make, 'model': model, 'logger': logger,
              'external_gps_data': external_gps_data,
              'internal_gps_data': internal_gps_data, 'ldmap': ldmap,
              }

    # Hacky but meh... maybe I should just pass args around
    sig = inspect.signature(process_video)
    argdict = vars(args)
    kwargs |= {arg: argdict[arg] for arg in sig.parameters
               if arg not in kwargs and arg in argdict}
    if set(sig.parameters) - set(kwargs) != {'input_ts_file'}:
        logger.error('Missing arguments:', set(sig.parameters)-set(kwargs))
        return

    frames_saved = {}
    if args.parallel == 1:
        cgitb.enable(format='text')
        frames_saved = {filename: process_video(filename, **kwargs)
                        for filename in inputfiles}

        missing = [fname for fname, count in frames_saved.items() if not count]
        if missing:
            logger.warning('Files with no frames saved:\n' +
                           os.linesep.join(str(x) for x in missing))
        return

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        threads = {executor.submit(process_video_with_exceptions,
                                   filename, **kwargs): filename
                   for filename in inputfiles}

        frames_saved = {threads[future]: future.result() for future
                        in concurrent.futures.as_completed(threads)}

        missing = sorted(fname for fname, count in frames_saved.items()
                         if not count)
        if missing:
            logger.warning('Files with no frames saved:\n' +
                           os.linesep.join(str(x) for x in missing))


if __name__ == '__main__':
    main()
