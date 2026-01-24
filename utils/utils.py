"""
@Author ：hhx
@Description ：Some basic functions
"""

from osgeo import gdal
import numpy as np
import pandas as pd
from tqdm import tqdm




def readGeoTIFF(fileName):
    """Read GeoTiff"""
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "file cannot be opened")
    im_width = dataset.RasterXSize  # columns of the raster
    im_height = dataset.RasterYSize  # rows of the raster
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    im_geotrans = dataset.GetGeoTransform()  # GeoTransform
    im_proj = dataset.GetProjection()  # Projection
    return im_data, im_geotrans, im_proj


def CreateGeoTiff(outRaster, image, geo_transform, projection):
    """Write GeoTiff"""
    no_bands = 0
    rows = 0
    cols = 0
    driver = gdal.GetDriverByName('GTiff')
    if len(image.shape) == 2:
        no_bands = 1
        rows, cols = image.shape
    elif len(image.shape) == 3:
        no_bands, rows, cols = image.shape

    DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Float32)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)
    if no_bands == 1:
        DataSet.GetRasterBand(1).WriteArray(image)
    else:
        for i in range(no_bands):
            DataSet.GetRasterBand(i + 1).WriteArray(image[i])
    del DataSet


def Median_filtering(image, window_size=3):
    high, wide = image.shape
    img = image.copy()
    mid = (window_size - 1) // 2
    med_arry = []
    for i in range(high - window_size):
        for j in range(wide - window_size):
            for m1 in range(window_size):
                for m2 in range(window_size):
                    med_arry.append(int(image[i + m1, j + m2]))
            med_arry.sort()  # Sort window pixels
            # Assign the median value of the filter window to the pixel in the middle of the filter window
            img[i + mid, j + mid] = med_arry[(len(med_arry) + 1) // 2]
            del med_arry[:]
    return img


def block_fn(x, center_val):
    unique_elements, counts_elements = np.unique(x.ravel(), return_counts=True)

    if np.isnan(center_val):
        return np.nan
    elif center_val == 1:
        return 1.0
    else:
        return unique_elements[np.argmax(counts_elements)]


def Majority_filter(x, window_size=3, type='spatial'):
    # Odd block sizes only  ( ? )
    if type == 'spatial':
        block_size = (window_size, window_size)
    else:
        block_size = (1, window_size)
    assert (block_size[0] % 2 != 0 and block_size[1] % 2 != 0)
    yy = int((block_size[0] - 1) / 2)
    xx = int((block_size[1] - 1) / 2)
    output = np.zeros_like(x)
    for i in range(0, x.shape[0]):
        miny, maxy = max(0, i - yy), min(x.shape[0] - 1, i + yy)
        for j in range(0, x.shape[1]):
            minx, maxx = max(0, j - xx), min(x.shape[1] - 1, j + xx)
            # Extract block to take majority filter over
            block = x[miny:maxy + 1, minx:maxx + 1]
            output[i, j] = block_fn(block, center_val=x[i, j])
    return output


def FilteringSpatial(img, method='NoFilter', window_size=3):
    """Spatio consistency modification"""
    if method == 'NoFilter':
        return img
    elif method == 'Median':
        return Median_filtering(img, window_size)
    elif method == 'Majority':
        return Majority_filter(img, window_size, type='spatial')


def DetectChangepoinst(x):
    xx = x[0]
    id = np.where((xx[1:] - xx[:-1]) != 0)
    return id[0], np.append(xx[id], xx[-1])


def FilteringSeries(data, method='NoFilter', window_size=5, output_info=True):
    """Temporal consistency modification"""
    series = data[None, :]
    if method == 'NoFilter':
        changepoints, changetypes = DetectChangepoinst(series)
        return data, changepoints, changetypes
    elif method == 'Majority':
        res = Majority_filter(series, window_size, type='series')
        changepoints, changetypes = DetectChangepoinst(res)
        if output_info == False:
            return res
        return res, changepoints, changetypes


def mat2rgb(mat):
    """Grayscale Matrix Visualization"""
    rgblist = [[88, 184, 255], [25, 70, 31], [138, 208, 27], [222, 168, 128], [212, 67, 56], [255, 214, 156],
               [255, 222, 173], [255, 255, 255], [255, 255, 255]]
    mat = mat.astype('int8')
    return np.array(rgblist)[mat]
