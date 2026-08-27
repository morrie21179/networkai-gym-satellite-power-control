import osmium
from shapely import wkb, wkt
from shapely.geometry import Point
from pyproj import Geod
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

from .population import get_city_wikidata

# [plot geo](https://max-coding.medium.com/getting-administrative-boundaries-from-open-street-map-osm-using-pyosmium-9f108c34f86)
# make sure you put your own pbf data in "osm_data" folder !
# download site: https://download.geofabrik.de/europe/ukraine.html
DIR_NAME = os.path.dirname(__file__)
OSM_FILE = os.path.join(DIR_NAME, "osm_data/ukraine-latest.osm.pbf")
ADMIN_LEVEL = "4"
COUNTRY = "Ukraine"

# oms data in csv format
CSV_FILENAME = os.path.join(DIR_NAME, "out_oms_data_l4.csv")
POP_CSV_FILENAME = os.path.join(DIR_NAME, "pop_data/ukraine_city_pop.csv")

# easier to generate user in groups
N_USER_PER_GROUP = 1000


def merge_two_dicts(x:dict, y:dict):
    z = x.copy()
    z.update(y) # insert y into z
    return z


class AdminAreaHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)

        self.areas = []
        self.wkbfab = osmium.geom.WKBFactory()

    def area(self, a):
        if (
            "admin_level" in a.tags and
            "name" in a.tags and
            (a.tags["admin_level"] == ADMIN_LEVEL or a.tags["admin_level"] == "2")
        ):
            try:
                wkbshape = self.wkbfab.create_multipolygon(a)
                shapely_obj = wkb.loads(wkbshape, hex=True)
                shapely_obj.area
                area_obj = {"id": a.id, "geo": shapely_obj}
                area = merge_two_dicts(area_obj, a.tags)

                self.areas.append(area)
            except RuntimeError:
                return


def get_osm_data(filename:str):
    handler = AdminAreaHandler()
    # start data file processing
    handler.apply_file(OSM_FILE, locations=True, idx='flex_mem')

    df = pd.DataFrame(handler.areas)
    gdf = gpd.GeoDataFrame(df, geometry="geo")

    # save geodf as csv
    gdf.to_csv(filename)
    

def plot_gdf(gdf:gpd.GeoDataFrame, level_num:int, 
            city_pop_dict:dict=None, users_per_oblast_dict:dict=None):
    """output a map png""" 
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes()

    # country boundary
    # EPSG 4326 CRS
    gdf[(gdf.admin_level == 2)].set_crs(crs=4326).plot(ax=ax, alpha=1, edgecolor="#000", linewidth=2) 

    # admin level boundaries
    # if ("ISO3166-2" in gdf.columns):
    #     admin_level_gdf = gdf[((gdf.admin_level==level_num) & (~gdf["ISO3166-2"].isna()))].set_crs(crs=4326)
    # else:
    admin_level_gdf = gdf[(gdf.admin_level==level_num)].set_crs(crs=4326)

    # add labels if provision
    if (level_num == 4):
        admin_level_gdf.plot(ax=ax, alpha=0.1, facecolor='b', edgecolor="#000", linewidth=1)
        for _, row in admin_level_gdf.iterrows():
            ann_txt = row["name:en"]
            # add pop number if provided dict
            # if (ann_txt in city_pop_dict):
            #     ann_txt += '\n' + str(city_pop_dict[ann_txt])
            ax.annotate(text=ann_txt, 
                        xy=(row.geo.centroid.x, row.geo.centroid.y), 
                        horizontalalignment='center')

    # plot users if provided dicct
    if (users_per_oblast_dict is not None):
        for _, row in admin_level_gdf.iterrows():
            name_i = row["name:en"]
            if (name_i not in users_per_oblast_dict):
                continue

            users_list = users_per_oblast_dict[name_i] # list containing user coord.
            user_xs, user_ys = [], []
            for user in users_list:
                user_xs.append(user[0])
                user_ys.append(user[1])
                
            plt.scatter(user_xs, user_ys, s=0.1, c='red')
            print(f"{name_i} plot done")


    if (not os.path.isdir("map")):
        os.makedirs("map")
    plt.savefig(f"map/uk_map_level_{level_num}.png")
    plt.show()
        

def read_pop_from_csv(gdf:gpd.GeoDataFrame, pop_csv_filename:str) -> dict:
    """read pop data from csv downloaded from OpenDataSoft website (Ukraine)"""
    pop_df = pd.read_csv(pop_csv_filename, sep=';')
    city_pop_dict = {}

    city_gdf = gdf[gdf["admin_level"] == 7]
    city_gdf = city_gdf[~city_gdf["name:en"].isna()]

    for i in range(len(city_gdf)):
        name_i = city_gdf.iloc[i]["name:en"]
        name_i_first_word = name_i.split(" ")[0]

        pop_i_df = pop_df.loc[
            pop_df["ASCII Name"].isin([name_i, name_i_first_word])
        ]

        if (len(pop_i_df) == 0):
            continue

        # city_i = pop_i_df["ASCII Name"].values[0]
        id_i = city_gdf.iloc[i]["id"]
        pop_i_lst = pop_i_df["Population"].values
        pop_i = sum(pop_i_lst)

        city_pop_dict[id_i] = pop_i

    # print(city_pop_dict) # --> ukraine: 443
    return city_pop_dict


def read_pop_from_qwiki(gdf:gpd.GeoDataFrame) -> dict:
    """if not found in API query, dict value is -1"""
    name_en_pdf = gdf["name:en"]
    pop_oblast_dict = {}

    for name in name_en_pdf.iloc:
        out = get_city_wikidata(city=name, country=COUNTRY)
        if (out is not None):
            pop = int(out["population"]["value"])
        else:
            pop = -1

        print(name, ':', pop)
        pop_oblast_dict[name] = pop

    # save dict
    with open("pop_data/pop_oblast_dict.json", "w") as f:
        json.dump(pop_oblast_dict, f)

    return pop_oblast_dict


def generate_users(gdf:gpd.GeoDataFrame, pop_oblast_dict:dict, out_filename="user_per_oblast.json"):
    """
    input:
    - gdf: geodataframe
    - pop_oblast_dict: dict containing population for each oblast
    - `out_filename`: output json filename

    return:
    - dict -- 
    key is geo name;  
    value is list containing user coordinates within geo

    1. generate users that stay within polygons (each point has `N_USER_PER_GROUP` users)
    """
    points_per_geo = {}

    # calculate area using ellps
    for i in range(len(gdf)):
        geo_i = gdf.iloc[i]["geo"]
        name_i = gdf.iloc[i]["name:en"]
        pop_i = pop_oblast_dict[name_i]

        # Use epllipsoid to estimate area
        # geod = Geod(ellps="WGS84")
        # geod_area = abs(geod.geometry_area_perimeter(geo_i)[0])
        # print(name_i, ':', f"{geod_area:.3f} m^2")

        # get bound of region
        minx, miny, maxx, maxy = geo_i.bounds
        points = []
        pop_i = pop_i // N_USER_PER_GROUP

        # random generate
        while (len(points) < pop_i):
            rand_x = np.random.uniform(minx, maxx)
            rand_y = np.random.uniform(miny, maxy)
            random_user_point = Point(rand_x, rand_y)

            if (geo_i.contains(random_user_point)):
                points.append((rand_x, rand_y))

        points_per_geo[name_i] = points
        print(name_i, ':', len(points), 'users')

    # save result
    out_path = os.path.join(DIR_NAME, f"pop_data/{out_filename}")
    with open(out_path, "w") as f:
        json.dump(points_per_geo, f)

    return points_per_geo


def generate_users_uniform(gdf:gpd.GeoDataFrame, pop_oblast_dict:dict, out_filename="user_per_oblast_uniform.json"):
    """
    Generate user distribution but uniformly

    generate users that stay within polygons (each point has `N_USER_PER_GROUP` users)

    input:
    - gdf: geodataframe (only `admin_level == 2`)
    - pop_oblast_dict: dict containing population for each oblast
    - `out_filename`: output json filename

    return:
    - dict -- 
    key is geo name;  
    value is list containing user coordinates within geo 
    """
    points_per_geo = {}

    gdf = gdf[gdf["admin_level"] == 2]

    # count total population
    total_pop = 0
    for _, pop in pop_oblast_dict.items():
        total_pop += pop

    print("total pop:", total_pop)

    geo = gdf.iloc[0]["geo"]
    name = gdf.iloc[0]["name:en"]
    minx, miny, maxx, maxy = geo.bounds

    points = []
    pop_i = total_pop // N_USER_PER_GROUP

    # random generate
    while (len(points) < pop_i):
        rand_x = np.random.uniform(minx, maxx)
        rand_y = np.random.uniform(miny, maxy)
        random_user_point = Point(rand_x, rand_y)

        if (geo.contains(random_user_point)):
            points.append((rand_x, rand_y))

        # make sure it's alive
        if (len(points) > 0 and len(points) % 100 == 0):
            print(f"{len(points)}/{pop_i} processed")

    points_per_geo[name] = points

    # save result
    out_path = os.path.join(DIR_NAME, f"pop_data/{out_filename}")
    with open(out_path, "w") as f:
        json.dump(points_per_geo, f)

    return points_per_geo


def generate_users_poisson(gdf:gpd.GeoDataFrame, ctry_geo, pop_oblast_dict:dict, out_filename="user_per_oblast_poisson.json"):
    """
    Generate users in each oblast following Poisson distribution
    (lambda = mean point of oblast coordinates, size = area population)

    generate users that stay within polygons (each point has `N_USER_PER_GROUP` users)

    input:
    - gdf: geodataframe
    - `ctry_geo` : geo of whole country (for country boundary)
    - pop_oblast_dict: dict containing population for each oblast
    - `out_filename`: output json filename

    return:
    - dict -- 
    key is geo name;  
    value is list containing user coordinates within geo
    """
    points_per_geo = {}

    # calculate area using ellps
    for i in range(len(gdf)):
        geo_i = gdf.iloc[i]["geo"]
        name_i = gdf.iloc[i]["name:en"]
        pop_i = pop_oblast_dict[name_i]

        # get bound of region
        minx, miny, maxx, maxy = geo_i.bounds
        lamx, lamy = int((minx + maxx) / 2), int((miny + maxy) / 2)
        points = []
        pop_i = pop_i // N_USER_PER_GROUP

        # random generate
        while (len(points) < pop_i):
            rand_x = np.random.poisson(lam=lamx, size=1)[0]
            rand_x = int(rand_x)
            rand_y = np.random.poisson(lam=lamy, size=1)[0]
            rand_y = int(rand_y)
            random_user_point = Point(rand_x, rand_y)

            if (ctry_geo.contains(random_user_point)):
                points.append((rand_x, rand_y))

        points_per_geo[name_i] = points
        print(name_i, ':', len(points), 'users')

    # save result
    out_path = os.path.join(DIR_NAME, f"pop_data/{out_filename}")

    with open(out_path, "w") as f:
        json.dump(points_per_geo, f)

    return points_per_geo


def calc_dist_lat_lon(lat1:float, lon1:float, lat2:float, lon2:float):
    """
    calculate distance between two points (provided their latitude and longtitude)

    [ref](https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters)
    """
    R = 6371 # earth radius in km
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(dLat / 2) ** 2 + \
        np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * \
        np.sin(dLon / 2) ** 2
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000 # in meters

def calc_cart_coord(lat:float, lon:float):
    """
    convert lat-lon system into Carterian

    [ref](https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates)
    """
    R = 6371_000 # earth radius in meters
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = R * np.cos(lat_r) * np.cos(lon_r)
    # y = R * np.cos(lat_r) * np.sin(lon_r)
    y = R * np.cos(lon_r) * np.sin(lat_r)
    # z = R *sin(lat_r)
    return (x, y)

def get_cart():
    """
    returns:
    - boundry of OSM map (min, max carterian coordinate on map)
    """
    # obtain osm data
    # ref: https://stackoverflow.com/questions/61122875/geopandas-how-to-read-a-csv-and-convert-to-a-geopandas-dataframe-with-polygons
    # get_osm_data(CSV_FILENAME)
    
    # read csv and convert to GeoDataFrame
    df = pd.read_csv(CSV_FILENAME)
    df["geo"] = df["geo"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geo", crs='epsg:4326')

    # get oblast data out of whole
    oblast_gdf = gdf[gdf["admin_level"] == 4]
    # drop NAN on name:en column
    oblast_gdf = oblast_gdf[oblast_gdf["name:en"].notna()]
    oblast_gdf = oblast_gdf.drop_duplicates(subset='name:en', keep='last') 
    
    # get population from json
    # with open("pop_data/pop_oblast_dict.json") as f:
    #     pop_oblast_dict = json.load(f) 
    # if you want to regenerate users, enable `read_pop_from_qwiki` instead
    # missing "Zaporizhia Oblast": 1638462 (from wiki 2022)  
    # read_pop_from_qwiki(oblast_gdf)


    ######## choose your option for user distribution ###########
    opt = 3
    #############################################################

    ctry_geo = gdf[gdf["admin_level"] == 2]["geo"]

    # Option 1: generate users in each oblast (in lat, lon)
    if (opt == 1):
        plot_level = 4
        user_per_oblast_json_filename = "user_per_oblast.json"
        # users_per_oblast_dict = generate_users(oblast_gdf, pop_oblast_dict, user_per_oblast_json_filename)

    # Option 2: generate users uniformly in Ukraine
    elif (opt == 2):
        plot_level = 2
        user_per_oblast_json_filename = "user_per_oblast_uniform.json"
        # users_per_oblast_dict = generate_users_uniform(df, pop_oblast_dict, user_per_oblast_json_filename)
        
    # Option 3: generate users in Poisson distribution
    else:
        plot_level = 4
        # get boundry of whole country
        user_per_oblast_json_filename = "user_per_oblast_poisson.json"
        # users_per_oblast_dict = generate_users_poisson(oblast_gdf, ctry_geo.values, pop_oblast_dict, user_per_oblast_json_filename)
    
    oblast_json_path = os.path.join(DIR_NAME, "pop_data" , user_per_oblast_json_filename)
    with open(oblast_json_path) as f:
        users_per_oblast_dict = json.load(f)

    # now transform user coord into Cartesian
    users_cart_dict = {}
    # base corrd is set to min x and y, of the country geo
    ctry_min_x, ctry_min_y, ctry_max_x, ctry_max_y = ctry_geo.bounds.values[0]
    base_cart = calc_cart_coord(ctry_min_x, ctry_min_y)
    max_cart = calc_cart_coord(ctry_max_x, ctry_max_y)

    for oblast_name in users_per_oblast_dict:
        user_coords = users_per_oblast_dict[oblast_name]
        new_coords = []

        for coord in user_coords:
            new_cart = calc_cart_coord(coord[0], coord[1])
            new_coords.append((new_cart[0], new_cart[1]))

        users_cart_dict[oblast_name] = new_coords

    # save cart coord
    post_str = user_per_oblast_json_filename[len("user_per_oblast_"):-5]
    save_coord_path = os.path.join(DIR_NAME, f"pop_data/user_cart_dict_{post_str}.json")
    with open(save_coord_path, "w") as f:
        json.dump(users_cart_dict, f)

    # plot
    # plot_gdf(gdf, plot_level, users_per_oblast_dict=users_per_oblast_dict)

    return base_cart, max_cart

if __name__ == '__main__':
    get_cart()

