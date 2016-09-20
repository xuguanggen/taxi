#! /usr/bin/env python

import numpy as np
import math as Math


def rad(d):
    return d * Math.pi /180.0


###  return hdistance unit is miles ################
def compute_hdistance(lon1,lat1,lon2,lat2):
    lon1 = float(lon1)
    lat1 = float(lat1)
    lon2 = float(lon2)
    lat2 = float(lat2)

    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lon1) - rad(lon2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) + 
        Math.cos(radLat1) * Math.cos(radLat2) * Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s


### return odistance ############################ 
def compute_odistance(lon1,lat1,lon2,lat2):
    return Math.sqrt((lon1 - lon2)*(lon1 - lon2) + (lat1 - lat2)*(lat1 - lat2))

