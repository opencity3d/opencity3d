from copy import deepcopy
import open3d as o3d
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import utm
import json
import pyproj
from scipy.interpolate import LinearNDInterpolator


# def convert_to_lat_lon(pcd, lat_min, lat_max, lon_min, lon_max):
#     points = np.asarray(pcd.points)
#     lon_max_geo, lat_max_geo, zone_number, zone_letter = utm.from_latlon(lat_max, lon_max)
#     lon_min_geo, lat_min_geo, zone_number, zone_letter = utm.from_latlon(lat_min, lon_min)
#     pcd_lat, pcd_lon = -points[:, 2], points[:, 0]
#     pcd_lat_geo = (pcd_lat - pcd_lat.min()) / (pcd_lat.max() - pcd_lat.min()) * (lat_max_geo - lat_min_geo) + lat_min_geo
#     pcd_lon_geo = (pcd_lon - pcd_lon.min()) / (pcd_lon.max() - pcd_lon.min()) * (lon_max_geo - lon_min_geo) + lon_min_geo
#     pcd_lat_, pcd_lon_ = utm.to_latlon(pcd_lon_geo, pcd_lat_geo, zone_number, zone_letter)
#     return pcd_lat_, pcd_lon_
def convert_to_lat_lon(pcd, coord_ref, type="mercator"):
    if type == "interpolate":
        lat_min, lat_max, lon_min, lon_max = coord_ref
        points = np.asarray(pcd.points)
        lon_max_geo, lat_max_geo, zone_number, zone_letter = utm.from_latlon(lat_max, lon_max)
        lon_min_geo, lat_min_geo, zone_number, zone_letter = utm.from_latlon(lat_min, lon_min)
        pcd_lat, pcd_lon = -points[:, 2], points[:, 0]
        pcd_lat_geo = (pcd_lat - pcd_lat.min()) / (pcd_lat.max() - pcd_lat.min()) * (lat_max_geo - lat_min_geo) + lat_min_geo
        pcd_lon_geo = (pcd_lon - pcd_lon.min()) / (pcd_lon.max() - pcd_lon.min()) * (lon_max_geo - lon_min_geo) + lon_min_geo
        pcd_lat_, pcd_lon_ = utm.to_latlon(pcd_lon_geo, pcd_lat_geo, zone_number, zone_letter)
        pcd_lat_ = (pcd_lat - pcd_lat.min()) / (pcd_lat.max() - pcd_lat.min()) * (lat_max - lat_min) + lat_min
        pcd_lon_ = (pcd_lon - pcd_lon.min()) / (pcd_lon.max() - pcd_lon.min()) * (lon_max - lon_min) + lon_min
        return pcd_lat_, pcd_lon_
    else:
        transverse_mercator_latitude, transverse_mercator_longitude = coord_ref
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_proj4(
                f"+proj=tmerc +lat_0={transverse_mercator_latitude} +lon_0={transverse_mercator_longitude}"
            ),
            pyproj.CRS.from_epsg(4326),
        )
        points = np.asarray(pcd.points)
        pcd_lat, pcd_lon = -points[:, 2], points[:, 0]
        latitude_degrees, longitude_degrees = transformer.transform(
            pcd_lon, pcd_lat
        )
        return latitude_degrees, longitude_degrees


def load_pcd_and_features(tag, crop_type):
    path = f"../data/embedded_point_clouds/{tag}"
    pcd = o3d.io.read_point_cloud(os.path.join(path, "generated_point_cloud.ply"))
    features = np.load(os.path.join(path, f"point_features_{crop_type}.npy")).astype(np.float16)
    features = features.transpose(1, 0, 2)
    print(features.shape, np.asarray(pcd.points).shape)
    assert len(features) == len(pcd.points)
    try:
        coord_ref = json.load(open(os.path.join(path, "coord_ref.json")))
        if "mercator" in coord_ref:
            coord_ref = coord_ref["mercator"]["transverse_mercator_latitude"], coord_ref["mercator"]["transverse_mercator_longitude"]
            latitude_degrees, longitude_degrees = convert_to_lat_lon(pcd, coord_ref, type="mercator")
        else:
            coord_ref = coord_ref["bounds_deg"]["lat_min"], coord_ref["bounds_deg"]["lat_max"], coord_ref["bounds_deg"]["lon_min"], coord_ref["bounds_deg"]["lon_max"]
            latitude_degrees, longitude_degrees = convert_to_lat_lon(pcd, coord_ref, type="interpolate")
    except:
        print("No coord ref found") 
        latitude_degrees, longitude_degrees = None, None
    return pcd, features, latitude_degrees, longitude_degrees

def _resample_to_regular_grid(x, y, features, n_points=1000):
    interp = LinearNDInterpolator(np.stack([x, y], axis=1), features)
    resampled_lat, resampled_lon = np.linspace(x.min(), x.max(), n_points), np.linspace(y.min(), y.max(), n_points)
    resampled_lat, resampled_lon = np.meshgrid(resampled_lat, resampled_lon)
    resampled_features = interp(resampled_lat, resampled_lon)
    return resampled_lat.flatten(), resampled_lon.flatten(), resampled_features.flatten()

def coord_to_geopandas(pcd_lat, pcd_lon, features, resample_to_regular_grid=False):
    if resample_to_regular_grid:
        old_features = deepcopy(features)
        for k, v in old_features.items():
            pcd_lat, pcd_lon, features[k] = _resample_to_regular_grid(pcd_lat, pcd_lon, v, n_points=1000 if features[k][0].size < 500 else 500 )
    features.update(**{"long": pcd_lon, "lat": pcd_lat})
    pcd_points_gpd = gpd.GeoDataFrame(features)
    pcd_points_gpd["geometry"] = pcd_points_gpd.apply(lambda row: Point(row['long'], row['lat']), axis=1)
    predicted_crime_geo_data = gpd.GeoDataFrame(pcd_points_gpd, geometry='geometry')
    predicted_crime_geo_data.set_crs(epsg=4326, inplace=True)
    return predicted_crime_geo_data


def aggregate_by_region(region_gdf, points, prefix="", project_on="points"):
    # crime_street_joined = gpd.sjoin(points, caba_rc, how='left', op='within')
    if project_on == "regions":
        crime_street_joined_predicted = gpd.sjoin(points, region_gdf, how='left', op='within')
        grouped = crime_street_joined_predicted.groupby('index_right')
        
        averages = grouped.mean()
        averages["n_points"] = grouped.size()
        for col in list(filter(lambda x: x not in ["long", "lat", "geometry"], points.columns)) + ["n_points"]:
            if col in averages.columns:
                col_vals = [averages[col].get(i, 0) for i in region_gdf.index] # sort by caba_rc index
                new_name = prefix + "avg_" + col if col != "n_points" else prefix + col
                region_gdf[new_name] = col_vals
                region_gdf[new_name] = region_gdf[new_name].fillna(0)
            else:
                region_gdf[prefix + col] = grouped[col].apply(list)
        return region_gdf
    elif project_on == "points":
        averages = gpd.sjoin(points, region_gdf, how='left', op='within')
        averages[prefix+"n_points"] = np.ones(len(averages))
        return averages