#For this: https://forum.mapillary.com/t/blueskysea-b4k-viofo-a119v3-and-mapillary
#Usage: python ts_processor.py --input 20210311080720_000421.TS  --sampling_interval 0.5 --folder output

import argparse
# import base64
import cgitb
import concurrent.futures
import configparser
import contextlib
import csv
import glob
import inspect
import io
import itertools
import logging
import math
import mmap
import multiprocessing
import os
import re
import struct
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timezone, timedelta, tzinfo
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Optional

import cv2
import piexif
import pyproj
from pymp4.parser import Box
from PIL import Image

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

THUMB_SIZE = 200
THUMB_QUALITY = 'web_low'

PIL_SAVE_SETTINGS = dict(quality=90,
                         # 4:2:0;  Source subsampling isn't better
                         subsampling=2,
                         optimize=True,
                         progressive=True)

EXTENSIONS = {'jpeg': 'jpg'}

JPEG_SETTINGS = (cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                 cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
                 cv2.IMWRITE_JPEG_QUALITY, 90,
                 # cv2.IMWRITE_JPEG_CHROMA_QUALITY, 80,
                 )

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
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def to_deg(value, loc):  #From here: https://gist.github.com/c060604
    """convert decimal coordinates into degrees, munutes and seconds tuple
    Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
    return: tuple like (25, 13, 48.343 ,'N')
    """
    loc_value = loc[0] if value < 0 else loc[1]

    degrees, minutes = divmod(abs(value), 1)
    minutes, seconds = divmod(minutes*60, 1)

    return (int(degrees), int(minutes), round(seconds*60, 5), loc_value)
    # deg =  int(abs_value)
    # t1 = (abs_value-deg)*60
    # minutes = int(t1)
    # sec = round((t1 - minutes)* 60, 5)
    # return (deg, minutes, sec, loc_value)

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
    side_long, side_short = (w, h) if width_is_longer else (h,w)

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
    """convert a number to rantional
    Keyword arguments: number
    return: tuple like (1, 2), (numerator, denominator)
    """
    f = Fraction(number).limit_denominator()
    return (f.numerator, f.denominator)

EXIF_MAPPER = {
    'max aperature': (piexif.ExifIFD.MaxApertureValue,
                      change_to_rational),
    'focal length': (piexif.ExifIFD.FocalLength, change_to_rational),
    '35mm equivalent focal length': (piexif.ExifIFD.FocalLengthIn35mmFilm, int),
    'focal plane x resolution': (piexif.ExifIFD.FocalPlaneXResolution, change_to_rational),
    'focal plane y resolution': (piexif.ExifIFD.FocalPlaneYResolution, change_to_rational),
}

# A129 Pro
# 7.2mm diagonal - crop factor of ~6 - 6.28 x 3.53mm
# Pixel size 1.62 µm x 1.62 µm

def build_exif_data(lat, lng, bear, make, model, datetm,
                    width=None, height=None, speed=None,
                    exifdata=None, thumbnail=None):
    """Adds GPS position as EXIF metadata
    Keyword arguments:
    lat -- latitude (as float)
    lng -- longitude (as float)
    """
    if not exifdata:
        exifdata = {}
    
    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])
    
    exiv_lat = change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2])
    exiv_lng = change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2])
    nbear = change_to_rational(round(bear, 2))
    dtstr = datetm.strftime(DATETIME_STR_FORMAT)
    
    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
        piexif.GPSIFD.GPSLatitude: exiv_lat,
        piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
        piexif.GPSIFD.GPSLongitude: exiv_lng,
        piexif.GPSIFD.GPSImgDirection: nbear,
        piexif.GPSIFD.GPSImgDirectionRef: 'T',
        piexif.GPSIFD.GPSMapDatum: 'WGS84',
    }

    if speed is not None:
        gps_ifd[piexif.GPSIFD.GPSSpeed] = change_to_rational(
            round(speed*3.6, 1))
        gps_ifd[piexif.GPSIFD.GPSSpeedRef] = 'K'

    zeroth_ifd = {
        piexif.ImageIFD.Make: make,
        piexif.ImageIFD.Model: model,
    }

    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: dtstr,
        piexif.ExifIFD.SceneType: b'\x01', # Directly photographed
        piexif.ExifIFD.FileSource: b'\x03', # Digital camera
        piexif.ExifIFD.SubjectDistanceRange: 3, # Distant subject
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


def lonlat_metric(xlon, xlat):
    return Mercator(xlon, xlat)

    # mx = xlon * (2 * math.pi * 6378137 / 2.0) / 180.0
    # my = math.log( math.tan((90 + xlat) * math.pi / 360.0 )) / (math.pi / 180.0)

    # my = my * (2 * math.pi * 6378137 / 2.0) / 180.0
    # return mx, my


def metric_lonlat(xmx, ymy):
    return Mercator(xmx, ymy, inverse=True)

    # xlon = xmx / (2 * math.pi * 6378137 / 2.0) * 180.0
    # xlat = ymy / (2 * math.pi * 6378137 / 2.0) * 180.0

    # xlat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)
    # return xlon, xlat


def decode_novatek_gps_packet(packet: bytes, tz: tzinfo=timezone.utc,
                              logger=logging) -> Optional[dict]:
    # Different units and firmware seem to put the data in different places
    if m := re.search(rb'[NS][EW]', packet):
        offset = m.start()
        input_packet = packet[offset-25:]
    else:
        logger.debug('Unknown packet: %s', packet[:188])
        logger.debug(packet[:188].hex(' ', 8))
        return None

    unpacked = gps_struct.unpack_from(input_packet)
    (hour, minute, second, year, month, day,
     active, lathem, lonhem, enclat, enclon, speed_knots,
     bearing) = unpacked

    if active != b'A':
        logger.debug('Not active.')
        logger.debug(unpacked)
        return None

    # Decode Novatek coordinate format to decimal degrees
    lat = fix_coordinates(lathem, enclat)
    lon = fix_coordinates(lonhem, enclon)

    # Validity checks
    if (not (0 <= hour <= 23) or not (0 <= minute <= 59) or
        not (0 <= second <= 60) or not (1 <= month <= 12) or
        not (1 <= day <= 31) or not (-180 <= lon <= 180) or
        not (-90 <= lat <= 90) or not (0 <= bearing <= 360) or
        speed_knots < 0):
        # Packet is invalid
        logger.debug('Packet invalid: %s', packet)
        logger.debug(packet.hex(' '))
        logger.debug(unpacked)
        return None

    speed = speed_knots * KNOTS_TO_MPS
    mx, my = lonlat_metric(lon, lat)
    ts = datetime(year=2000+year, month=month, day=day, hour=hour,
                  minute=minute, second=second, tzinfo=tz)

    return dict(lat=lat, latR=lathem, lon=lon, lonR=lonhem,
                bearing=bearing, speed=speed, mx=mx, my=my,
                metric=0, prevdist=0, ts=ts)


def detect_file_type(input_file, device_override='', logger=logging):
    device = "X"
    make = "unknown"
    model = "unknown"
    detected_offsets = []

    p = Path(input_file)
    extension = p.suffix.lower()
    known = extension in ('.ts', '.mp4') # Try everything if extension unknown
    
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
        with open(input_file, "rb") as fx:
            with mmap.mmap(fx.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.madvise(mmap.MADV_RANDOM)
                eof = mm.size()
                while mm.tell() < eof:
                    lazybox = struct.unpack('!I 4s',
                                            mm[mm.tell():mm.tell()+8])
                    # Save some parsing overhead since we don't care...
                    if lazybox[1] in (b'ftyp', b'mdat', b'skip') and lazybox[0] > 1:
                        mm.seek(lazybox[0], os.SEEK_CUR)
                        continue
                    
                    try:
                        box = Box.parse_stream(mm)
                    except:
                        pass
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
                    elif box.type == b"gps": #has Novatek-specific stuff
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


def get_gps_data_nt(input_ts_file, device, tz, logger=logging, offsets=None):
    packetno = 0
    locdata = {}
    with open(input_ts_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            for (offset, length) in offsets:
                input_packet = mm[offset:offset+length]
                pos = input_packet.find(b'freeGPS')
                if pos < 0:
                    #print('?bogus packet at', offset)
                    continue
                else:
                    input_packet = input_packet[pos:]
                
                if currentdata := decode_novatek_gps_packet(
                        input_packet, tz, logger):
                    locdata[packetno] = currentdata
                    packetno += 1
            if packetno > 0:
                return locdata, packetno
                
            for match in re.finditer(b'freeGPS', mm[offset:]):
                print('packet at', match.start())
                input_packet = mm[match.start():match.start()+188]
                if currentdata := decode_novatek_gps_packet(
                        input_packet, tz, logger):
                    locdata[packetno] = currentdata
                    packetno += 1

    return locdata, packetno

garmin_gps_struct = struct.Struct('>s xx i i')

def get_gps_data_garmin (input_ts_file, device, tz):
    packetno = 0
    locdata = {}
    
    with open(input_ts_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            for match in re.finditer(b'\x00\x14\x50\x4E\x44\x4D\x00\x00\x00\x00', mm):
                input_packet = mm[match.start():match.start()+56]
                speed_kmh, lat, lon = garmin_gps_struct.unpack_from(input_packet)
                lat /= 11930464.711111112
                lon /= 11930464.711111112
                mx, my = lonlat_metric(lon, lat)
                
                locdata[packetno] = dict(
                    speed=speed_kmh / 3.6,
                    bearing=0, ts=datetime.fromtimestamp(0, timezone.utc),
                    lat=lat, lon=lon, mx=mx, my=my,
                    latR=b'N' if lat >= 0 else b'S',
                    lonR=b'E' if lon >= 0 else b'W',
                    prevdist=0, metric=0)
                packetno += 1
                
    return locdata, packetno


def parse_nmea_rmc(line: bytes, tzone='Z') -> dict:
    bits = line.split(b',')
    datestr = f'{bits[9]} {bits[1]}{tzone}'
    ts = datetime.strptime(datestr, '%d%m%y %H%M%S.%f%z')

    rawlat, lathem, rawlon, lonhem = bits[3:7]
    lat = fix_coordinates(lathem, rawlat)
    lon = fix_coordinates(lonhem, rawlon)
    mx, my = lonlat_metric(lon, lat)

    return dict(ts=ts, active=bits[2], lat=lat, latR=lathem, lon=lon,
                lonR=lonhem, mx=mx, my=my, speed=float(bits[7])*KNOTS_TO_MPS,
                bearing=float(bits[8]), metric=0, prevdist=0)
                                    

def get_gps_data_nmea(input_file, device, tz):
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
                        #print (inp.type.decode("utf-8"))
                        if inp.type == b"gps": #NMEA-based
                            lines = inp.data
                            for line in lines.splitlines():
                                if b"$GPRMC" in line:
                                    currentdata = parse_nmea_rmc(line, tzstr)
                                    ts = currentdata['ts']
                                    active = currentdata['active']
                                    if active == b'A' and ts > prevts:
                                        locdata[packetno] = currentdata
                                        prevts = ts
                                        packetno += 1
                        offset += inp.end
            
    return locdata, packetno


def get_gps_data_ts (input_ts_file, device, tz, logger=logging):
    packetno = 0
    locdata = {}
    prevdata = {}
    # prevpacket = None
    with open(input_ts_file, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            mm.madvise(mmap.MADV_SEQUENTIAL)
            input_packet = mm.read(188) #First packet, try to autodetect

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


def build_gpxtree(locdata: dict, make: str, model: str,
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
        # comment.text = f'track point {i}'

        # if thumb := fix.get('thumbnail'):
        #     encthumb = base64.b64encode(thumb).decode("us-ascii")
        #     url = ET.SubElement(trkpt, 'url')
        #     url.text = f'data:image/jpeg;base64,{encthumb}'
            
        speed = ET.SubElement(trkpt, 'speed')
        speed.text = f"{fix['speed']:.1f}"

    bounds.set('minlat', f"{minlat:.6f}")
    bounds.set('maxlat', f"{maxlat:.6f}")
    bounds.set('minlon', f"{minlon:.6f}")
    bounds.set('maxlon', f"{maxlon:.6f}")
    metatime.text = maxtime.isoformat()
    return ET.ElementTree(gpx)


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

    return dict(
        ts=data1["ts"]-(data2["ts"]-data1["ts"]),
        mx=mx, my=my, lat=lat, lon=lon,
        bearing=(data1["bearing"]-deltabearing) % 360,
        speed=data1["speed"]-(data2["speed"]-data1["speed"]),
        metric=0, prevdist=0)


def interpolate_locdata(start: dict, end: dict, proportions: dict=None,
                        position: float=None) -> dict:
    deltamx = end["mx"]-start["mx"]
    deltamy = end["my"]-start["my"]
    deltats = end["ts"]-start["ts"]
    deltadist = end['metric']-start['metric']
    deltabearing = (end['bearing']-start['bearing']) % 360
    if deltabearing > 180:
        deltabearing -= 360

    deltaspeed = end['speed']-start['speed']

    if position is not None:
        proportions = {0 : position}

    locdata = {}
    for i, curpos in proportions.items():
        mx = start["mx"] + deltamx*curpos
        my = start["my"] + deltamy*curpos
        lon, lat = metric_lonlat(mx, my)
        # dist = WGS84.line_length(lats=(lat, start['lat']),
        #                          lons=(lon, start['lon']))
        dist = deltadist*curpos
        prevdist = start['prevdist'] + dist
        metric = start['metric'] + dist
        bearing = (start['bearing'] + curpos*deltabearing) % 360
        
        # if abs(deltabearing) > 2:
        #     print('here', curpos, deltabearing, curpos*deltabearing, bearing)
                
        locdata[i] = dict(
            ts=start["ts"] + deltats*curpos,
            mx=mx, my=my, lat=lat, lon=lon,
            speed=start["speed"] + deltaspeed*curpos,
            bearing=bearing, metric=metric, prevdist=prevdist)

    if position is not None:
        return locdata[0]
    return locdata


def process_video(input_ts_file: str, folder: str, thumbnails: bool=False,
                  mask=None, make='', model='', kml_out=False,
                  device_override='', tz: float=0,
                  crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,
                  sampling_interval: float=0.5, min_points=5,
                  min_speed=-1, timeshift: float=0, metric_distance: float=0,
                  turning_angle: float=0, suppress_cv2_warnings=False,
                  use_sampling_interval=True, config=None, sensor_width=None,
                  bearing_modifier: float=0, use_speed=False, limit: int=0,
                  max_aperature=None, rotate: float=0, output_format='jpeg',
                  focal_length=None, min_coverage=90,
                  keep_aspect_ratio=False,
                  csv_out=False, gpx_out=False, dry_run=False) -> int:
    logger = multiprocessing.get_logger()
    # logger = logging.getLogger('process_video.'+input_ts_file)
    logger.info('Processing %s', input_ts_file)
    device, detected_make, detected_model, offsets = detect_file_type(
        input_ts_file, device_override, logger)
    logger.debug('Detected %s %s %s', detected_make, detected_model, device)
    make = make or detected_make
    model = model or detected_model

    do_crop = any((crop_left, crop_top, crop_right, crop_bottom))

    vidcontext = (suppress_stdout_stderr if suppress_cv2_warnings else 
                  contextlib.nullcontext)

    rot_radians = math.radians(rotate)
    exifdata = {}
    modstr = f'{make} {model}'

    video = cv2.VideoCapture(str(input_ts_file))
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug('FPS: %d; LEN: %d', fps, length)

    if not config:
        config = configparser.ConfigParser()
    
    make = config.get(modstr, 'make', fallback=make)
    model = config.get(modstr, 'model', fallback=model)

    max_aperature = config.get(modstr, 'max_aperature',
                               fallback=max_aperature)
    if max_aperature:
        # Convert to APEX - # of stops from f/1
        apex_max_aperature = math.log(max_aperature**2, 2)
        # F numbers are rounded to 2 SF, so reverse the rounding
        # and store precisely
        apex_max_aperature = Fraction(round(apex_max_aperature)*6, 6)
        exifdata['max aperature'] = apex_max_aperature

    sensor_width = config.get(modstr, 'sensor_width',
                              fallback=sensor_width)
    if sensor_width:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pixel_width = sensor_width/width/10
        exifdata['focal plane x resolution'] = pixel_width
        exifdata['focal plane y resolution'] = pixel_width

    focal_length = config.get(modstr, 'focal_length', fallback=focal_length)
    if focal_length:
        exifdata['focal length'] = focal_length

    if sensor_width and focal_length:
        crop_factor = 35/sensor_width
        exifdata['35mm equivalent focal length'] = focal_length*crop_factor
            
    interval = int(sampling_interval*fps)
    if interval == 0:
        interval = 1
    locdata = {}
    packetno = 0

    tzone = timezone(timedelta(hours=tz))
    if device in "BVS":
        locdata, packetno = get_gps_data_ts(input_ts_file, device, tzone, logger)
    elif device in "T":
        locdata, packetno = get_gps_data_nt(input_ts_file, device, tzone,
                                            logger, offsets)
    elif device in "N":
        locdata, packetno = get_gps_data_nmea(input_ts_file, device, tzone)
    elif device in "G":
        locdata, packetno = get_gps_data_garmin(input_ts_file, device, tzone)

    logger.debug("GPS data analysis ended; %d points", len(locdata))
    if packetno:
        logger.debug("Frames per point: %f", length/packetno)

    if not locdata:
        logger.warning('No GPS data found in %s', input_ts_file)
        return 0
        
    ###Logging

    if not os.path.exists(folder) and not dry_run:
        os.makedirs(folder)
    
    fnbase, _ = os.path.splitext(os.path.join(folder, os.path.basename(input_ts_file)))
    fnbase += '_'
    if csv_out:
        with open(f'{fnbase}pre_interp.csv', 'w', newline='') as fd:
            csvfile = csv.DictWriter(fd, fieldnames=locdata[0],
                                     delimiter=';')
            csvfile.writeheader()
            csvfile.writerows(locdata.values())

    if gpx_out:
        gpxtree = build_gpxtree(locdata, make, model)
        gpxtree.write(f'{fnbase}pre_interp.gpx')
    ###

    # Remove any insane coordinate values
    droplist = []

    keylist = sorted(locdata)
    lats = [locdata[j]['lat'] for j in keylist]
    lons = [locdata[j]['lon'] for j in keylist]
    distances = WGS84.line_lengths(lats=lats, lons=lons)

    for i, prev_distance in enumerate(distances):
        # Probably garbage data if we've jumped > 10 km
        if prev_distance > 10000:
            logger.debug('Dropping (bogus?) point %d %.1f meters away', i,
                         prev_distance)
            droplist.append(i+1)
            
    for i in droplist:
        del locdata[i]

    # Need at least 2 points to interpolate
    if len(locdata) < 2:
        logger.info('Not enough GPS data for interpolation; need >= 2 points, got %d', len(locdata))
    elif len(locdata)<min_coverage*length*0.01/fps:
        logger.info("Not enough GPS data for interpolation; %d%% needed, %f%% found", min_coverage, 100*len(locdata)/length*fps)
    elif len(locdata)<length/fps:
        logger.info("Interpolating missing points")
        i = 0
        while i < length/fps:
            if i not in locdata:
                #Find previous existing
                prev_data = i - 1
                while prev_data not in locdata and prev_data > 0:
                    prev_data -= 1

                next_data = i + 1
                #Find next existing
                while next_data not in locdata and next_data < length/fps:
                    next_data += 1

                if prev_data in locdata and next_data in locdata:
                    gap = next_data - prev_data

                    props = {counter : (counter - prev_data)/gap
                             for counter in range(i, next_data)}
                    locdata.update(interpolate_locdata(
                        prev_data, next_data, props))
                    i = next_data

            i += 1

    assert locdata

    if len(locdata) >= 2:
        for i in range(min(locdata), -5, -1):
            if i not in locdata:
                locdata[i] = extrapolate_locdata(locdata[i+1], locdata[i+2])

        for i in range(max(locdata)+1, int(1.1 * length / fps)):
            if i not in locdata:
                locdata[i] = extrapolate_locdata(locdata[i-1], locdata[i-2])

        keylist = sorted(locdata)
        lats = [locdata[j]['lat'] for j in keylist if j >= 0]
        lons = [locdata[j]['lon'] for j in keylist if j >= 0]
        distances = WGS84.line_lengths(lats=lats, lons=lons)

        for i, prev_distance in enumerate(distances):
            locdata[i+1]['prevdist'] = prev_distance

        for i, cum_distance in enumerate(itertools.accumulate(distances)):
            locdata[i+1]['metric'] = cum_distance
        
    # for i in range(1, max(locdata):
    #     distance = WGS84.line_length(
    #         lats=(locdata[i]['lat'], locdata[i-1]['lat']),
    #         lons=(locdata[i]['lon'], locdata[i-1]['lon']))
    #     locdata[i]["prevdist"] = distance
    #     locdata[i]["metric"] = locdata[i-1]["metric"] + distance

    locdata_ts = {}
    for i, gps_loc in locdata.items():
        # print(i, gps_loc)
        clock = gps_loc['posix_clock'] = gps_loc['ts'].timestamp()
        locdata_ts[clock] = gps_loc
        
    ###Logging
    if csv_out and not dry_run:
        with open(f'{fnbase}post_interp.csv', 'w', newline='') as fd:
            csvfile = csv.DictWriter(fd, fieldnames=locdata[0],
                                     delimiter=';')
            csvfile.writeheader()
            csvfile.writerows(locdata.values())

    if gpx_out and not dry_run:
        gpxtree = build_gpxtree(locdata, make, model)
        gpxtree.write(f'{fnbase}post_interp.gpx')

    ###

    if len(locdata) < min_points:
        logger.warning("Not enough GPS data for frame extraction: %s",
                       input_ts_file)
        return

    if dry_run:
        logger.info("Would be extracting from %s now", input_ts_file)
    else:
        logger.info("Extraction started: %s", input_ts_file)

    framecount = 0
    count = 0
    success, image = video.read()
    turning = False

    # current_time = locdata[0]['ts']
    # framelength = 1/fps
    timestamps = sorted(locdata_ts)
    lastframe = {}
    useframe = True
    next_distance = 0
    raw_thumbnail = None

    photos = []
    while success and framecount < length and (not limit or count < limit):
        # Interpolate time and coordinates
        current_time = locdata[0]['posix_clock'] + (framecount/fps)

        prevts = [ts for ts in timestamps if ts <= current_time + timeshift]
        nextts = [ts for ts in timestamps if ts > current_time + timeshift]

        if not nextts:
            break
        
        prev_fix = max(prevts)
        next_fix = min(nextts)

        locprev = locdata_ts[prev_fix]
        locnext = locdata_ts[next_fix]
        
        currentpos = (current_time+timeshift-prev_fix)/(next_fix-prev_fix)
        posinfo = interpolate_locdata(locprev, locnext, position=currentpos)

        if framecount % 60 == 0:
            logger.debug('fr %d ts %.3f target %.3fm at %.3fm',
                         framecount, current_time, next_distance,
                         posinfo['metric'])
        # logger.debug('%.3f %.3f', next_distance, posinfo['metric'])
        # logger.debug(currentpos)
        # logger.debug(lastframe)
        # logger.debug(posinfo)
        
        if (use_sampling_interval and not lastframe or
            current_time-lastframe.get('posix_clock', 0) > sampling_interval):
            useframe = True
            
        if metric_distance and not turning_angle:
            if posinfo['metric'] < next_distance:
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
            if lastframe:
                last_distance = WGS84.line_length(
                    lats=(lastframe['lat'], posinfo['lat']),
                    lons=(lastframe['lon'], posinfo['lon']))
                travel_distance = posinfo['metric'] - lastframe['metric']
                speed = posinfo['speed']
                anglediff = (lastframe['bearing'] - posinfo['bearing']) % 360
                if last_distance < 1 or travel_distance < 1 or speed < 0.5:
                    framecount += 1
                    continue
                elif turning_angle < min(anglediff, 360-anglediff):
                    logger.debug('turning')
                    useframe = True
                    turning = True
                elif metric_distance and posinfo['metric'] >= next_distance:
                    useframe = True
                    turning = False
                # elif turning: # Capture end of turn
                #     useframe = True
                #     turning = False
                else:
                    framecount += 1
                    continue
                
                logger.debug('%d angle: %.3f dist: %.3f %.3f %.3f v: %.1f',
                             framecount, min(anglediff, 360-anglediff),
                             last_distance, travel_distance,
                             posinfo['metric'] - next_distance, speed)
            else:
                useframe = True
        
        if useframe:
            with vidcontext():
                video.set(1, framecount)
                success, image = video.read()
            
            if not success:
                break

            if mask:
                image = cv2.bitwise_and(image, image, mask=mask)

            pil_image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            width, height = pil_image.size
            if abs(rotate) % 90 == 0:
                pil_image = pil_image.rotate(rotate)
                width, height = pil_image.size
            elif rotate:
                orig_width, orig_height = width, height
                pil_image = pil_image.rotate(rotate, Image.BICUBIC,
                                             expand=True)
                width, height = pil_image.size
                nw, nh = rotatedRectWithMaxArea(width, height, rot_radians)

                if keep_aspect_ratio:
                    orig_ratio = orig_width / orig_height
                    new_ratio = nw / nh
                    if new_ratio > orig_ratio:
                        nw = nh*orig_ratio
                    else:
                        nh = nw/orig_ratio
                
                x1 = (width - nw)//2 + crop_left
                y1 = (height - nh)//2 + crop_top
                
                x2 = x1 + nw - crop_right
                y2 = y1 + nh - crop_bottom

                pil_image = pil_image.crop((x1, y1, x2, y2))
                width, height = pil_image.size
            elif do_crop:
                pil_image = pil_image.crop(
                    (crop_left, crop_top, width - crop_right,
                     height - crop_bottom))
                width, height = pil_image.size

            datetime_taken = posinfo['ts']
            # datetime_taken = datetime.fromtimestamp(new_ts + tz*3600)

            ext = EXTENSIONS.get(output_format.lower(), output_format.lower())
            
            jpgname = f'{fnbase}{count:06d}.{ext}'
            logger.debug('Would save %s' if dry_run else 'Saving %s', jpgname)

            # cv2.imwrite(jpgname, image, JPEG_SETTINGS)

            bearing = (posinfo['bearing'] + bearing_modifier) % 360

            if thumbnails:
                thumbnail = pil_image.copy()
                thumbnail.thumbnail((THUMB_SIZE, THUMB_SIZE))
                o = io.BytesIO()
                thumbnail.save(o, 'jpeg', quality=THUMB_QUALITY)
                raw_thumbnail = o.getvalue()
                # posinfo['thumbnail'] = raw_thumbnail
            
            exif_bytes = build_exif_data(
                posinfo['lat'], posinfo['lon'],
                bearing, make, model, datetime_taken,
                speed=(posinfo['speed'] if use_speed else None),
                width=width, height=height, exifdata=exifdata,
                thumbnail=raw_thumbnail)

            if not dry_run:
                pil_image.save(jpgname, output_format, **PIL_SAVE_SETTINGS,
                               exif=exif_bytes)

            posinfo['photo'] = jpgname
            next_distance = posinfo['metric'] + metric_distance
            lastframe = posinfo
            count += 1
            photos.append(posinfo)

        # End of loop
        if not metric_distance and not turning_angle:
            framecount += int(sampling_interval*fps)
            useframe = True
        else:
            framecount += 1
            useframe = False
        
    video.release()
    if dry_run:
        logger.info('%s processed; would have extracted %d image(s)',
                    input_ts_file, count)
        return 0

    logger.info('%s processed; %d image(s) extracted', input_ts_file, count)

    if gpx_out and not dry_run:
        gpxtree = build_gpxtree(photos, make, model, generate_track=False)
        gpxtree.write(f'{fnbase}photos.gpx')

    if kml_out and not dry_run:
        kmltree = build_kml(photos, make, model)
        kmltree.write(f'{fnbase}photos.kml')

    return count


def process_video_with_exceptions(infile: os.PathLike, folder: os.PathLike,
                                  *args, **kwargs) -> int:
    try:
        return process_video(infile, folder, *args, **kwargs)
    except Exception:
        tb = cgitb.text(sys.exc_info())
        print(tb, file=sys.stderr)
        raise
        

def find_files(filelist: list, recursive=False) -> list:
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
                               if VALID_FILENAME.match(str(f)))
        else:
            logger.error("Can't find input file: %s", fname)
            sys.exit(1)
            
    return inputfiles


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
    parser.add_argument('--csv', action='store_true', dest='csv_out',
                        help="create csv from coordinates before and after interpolation.")
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
    parser.add_argument('--max-aperature', action='store', type=Decimal,
                        help='camera maximum aperature (F number)')
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

    use_sampling_interval = (not args.metric_distance)

    mask = cv2.imread(args.maskfile, 0) if args.maskfile else None

    if args.dry_run:
        logger.warning('*** Dry run - no data saved ***')

    kwargs = {'config': config, 'mask': mask,
              'use_sampling_interval': use_sampling_interval}

    # Hacky but meh... maybe I should just pass args around
    sig = inspect.signature(process_video)
    argdict = vars(args)
    kwargs |= {arg: argdict[arg] for arg in sig.parameters if arg in argdict}
    if set(sig.parameters) - set(kwargs) != {'input_ts_file'}:
        log.error('Missing arguments:', set(sig.parameters)-set(kwargs))
        return

    if args.parallel == 1:
        cgitb.enable(format='text')
        for filename in inputfiles:
            process_video(filename, **kwargs)
        return

    threads = []
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        threads = [executor.submit(process_video_with_exceptions,
                                   filename, **kwargs)
                   for filename in inputfiles]
        
        for thread in threads:
            thread.result()
            if thread.exception():
                return
            

if __name__ == '__main__':
    main()
