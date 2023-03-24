import math

def calculate_center_coords(coords_list):
    lat_mean = sum([coords[0] for coords in coords_list]) / len(coords_list)
    lng_mean = sum([coords[1] for coords in coords_list])  / len(coords_list)

    return [round(lat_mean, 6), round(lng_mean, 6)]

def haversine_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 3956 * c  # radius of earth in miles

    return distance