import pandas as pd
import random
from datetime import datetime, timedelta

from ML.Evaluation._file_access_helper_functions import save_to_file

def dummy_create_queries():
    return {
        "where": ["2008-02-02 15:38:08"],
        "distance": [
            {
                "time_first": "2008-02-02 15:38:08",
                "time_last": "2008-02-02 15:58:08"
            },
            {
                "time_first": "2008-02-02 13:31:08",
                "time_last": "2008-02-02 13:31:08"
            }
        ],
        "when": [
            {  # 13:38, 15:41
                "longitude": 116.51230,
                "latitude": 39.92180
            },
            {  # 18:02:30
                "longitude": 116.595000,
                "latitude": 39.991000
            },
            {
                "longitude": 116.51230,
                "latitude": 39.991000
            }
        ],
        "how_long": [
            # 00:19, 00:01
            {
                "first_point": {
                    "longitude": 116.51230,
                    "latitude": 39.92180
                },
                "last_point": {
                    "longitude": 116.51372,
                    "latitude": 39.92323
                }
            }
        ],
        "count": [
            {
                "longitude": 116.51230,
                "latitude": 39.92180,
                "radius": 11
            },
            {
                "longitude": 116.244311,
                "latitude": 39.911225,
                "radius": 1000
            }
        ],
        # time_first < time_last
        "knn": [
            {
                "longitude": 116.51230,
                "latitude": 39.92180,
                "time_first": "2008-02-02 12:38:08",
                "time_last": "2008-02-02 17:58:08",
                "k": 3,
            },
            {
                "longitude": 116.244311,
                "latitude": 39.911225,
                "time_first": "2008-02-02 13:31:08",
                "time_last": "2008-02-02 13:31:09",
                "k": 5,
            },
        ],
        "window": [
            {
                "first_point":
                    {"longitude": 116.244311, "latitude": 39.911225},
                "last_point":
                    {"longitude": 116.51230, "latitude": 39.92180},
                "t1": "2008-02-02 12:38:00",
                "t2": "2008-02-02 17:58:09",
            },
            {
                "first_point":
                    {"longitude": 116.60000, "latitude": 39.99200},
                "last_point":
                    {"longitude": 116.60500, "latitude": 39.992500},
                "t1": "2008-02-02 12:38:00",
                "t2": "2008-02-02 19:00:00",
            }
        ]
    }


def create_queries(amount_of_individual_queries, version):
    print("Creating queries...")
    queries = {}

    queries["where"] = [
        1202103846,
        1202267637,
        1202365650,
        1202458674,
        1202312510,
        1202195900,
        1201980843,
        1202056464,
        1202342023,
        1202032928,
        1201974734,
        1202014061,
        1202190147,
        1202122414,
        1202028061
    ]

    queries["distance"] = [
        {'time_first': 1201979147, 'time_last': 1202167366},
        {'time_first': 1202059069, 'time_last': 1202305325},
        {'time_first': 1202101036, 'time_last': 1202409164},
        {'time_first': 1202073536, 'time_last': 1202166568},
        {'time_first': 1202009635, 'time_last': 1202489643},
        {'time_first': 1202134260, 'time_last': 1202299387},
        {'time_first': 1202296572, 'time_last': 1202523640},
        {'time_first': 1202131702, 'time_last': 1202224894},
        {'time_first': 1202184319, 'time_last': 1202452268},
        {'time_first': 1202147667, 'time_last': 1202185249},
        {'time_first': 1202036113, 'time_last': 1202512359},
        {'time_first': 1202321946, 'time_last': 1202324128},
        {'time_first': 1202033237, 'time_last': 1202061775},
        {'time_first': 1202006670, 'time_last': 1202393168},
        {'time_first': 1201984658, 'time_last': 1202466754},
    ]

    queries["when"] = [
        # All above 2500 results and most are around 5000
        {'latitude': 39.94399290299451, 'longitude': 116.3737908034472},
        {'latitude': 39.88840721797357, 'longitude': 116.38335893721406},
        {'latitude': 39.961488738769, 'longitude': 116.44082580217137},
        {'latitude': 39.95584882583082, 'longitude': 116.42815882220785},
        {'latitude': 39.857102766743026, 'longitude': 116.36683988301608},
        {'latitude': 39.89619186633144, 'longitude': 116.44772099170021},
        {'latitude': 39.86624181146118, 'longitude': 116.32145286089104},
        {'latitude': 39.885232693496995, 'longitude': 116.30272911998595},
        {'latitude': 39.90350554734829, 'longitude': 116.34691325694389},
        {'latitude': 39.90363229686544, 'longitude': 116.34386209997525},
        {'latitude': 39.943850592651195, 'longitude': 116.43841943304479},
        {'latitude': 39.91713903685549, 'longitude': 116.4339418058951},
        {'latitude': 39.95342455036681, 'longitude': 116.43388109496176},
        {'latitude': 39.86713886236004, 'longitude': 116.34585037575259},
        {'latitude': 39.906085969927446, 'longitude': 116.3531550661019},
    ]

    queries["how_long"] = [
        # All have around 100-200 results except the one with a comment after.
        {'first_point': {'latitude': 39.97286205848362, 'longitude': 116.52683259169358}, 'last_point': {'latitude': 39.89414046097204, 'longitude': 116.4168611780795}},
        {'first_point': {'latitude': 39.97717924347944, 'longitude': 116.40005584591823}, 'last_point': {'latitude': 39.89572731660192, 'longitude': 116.42705318078957}},
        {'first_point': {'latitude': 39.88085912637115, 'longitude': 116.3607557355413}, 'last_point': {'latitude': 39.92202807188227, 'longitude': 116.42545155551842}},
        {'first_point': {'latitude': 39.879816315497756, 'longitude': 116.43939378863415}, 'last_point': {'latitude': 40.020222413917914, 'longitude': 116.55529640785933}},
        {'first_point': {'latitude': 39.94741569600757, 'longitude': 116.52969982885067}, 'last_point': {'latitude': 40.05716569203948, 'longitude': 116.4078792418577}},
        {'first_point': {'latitude': 39.91150279928188, 'longitude': 116.31291294739735}, 'last_point': {'latitude': 39.958394709458034, 'longitude': 116.31137227889101}},
        {'first_point': {'latitude': 40.002123299437564, 'longitude': 116.35442344303735}, 'last_point': {'latitude': 39.96218921368474, 'longitude': 116.47288369528724}},
        {'first_point': {'latitude': 39.892327689603455, 'longitude': 116.41838246261867}, 'last_point': {'latitude': 39.88993071580905, 'longitude': 116.22251352826696}},
        {'first_point': {'latitude': 39.83678817344769, 'longitude': 116.29093794450961}, 'last_point': {'latitude': 39.91849904192865, 'longitude': 116.35164652362936}},
        {'first_point': {'latitude': 39.863281708030655, 'longitude': 116.40220084557991}, 'last_point': {'latitude': 40.08443195293111, 'longitude': 116.54986920387836}},
        {'first_point': {'latitude': 39.96822440572677, 'longitude': 116.46130129237741}, 'last_point': {'latitude': 39.96821029565072, 'longitude': 116.42756474870329}}, # 2610 results
        {'first_point': {'latitude': 39.86661100017184, 'longitude': 116.56925096614546}, 'last_point': {'latitude': 39.91260450389531, 'longitude': 116.33572791967849}},
        {'first_point': {'latitude': 40.00027563348961, 'longitude': 116.32576547766183}, 'last_point': {'latitude': 39.91907672948717, 'longitude': 116.3595929701036}},
        {'first_point': {'latitude': 39.94973035386802, 'longitude': 116.25801546768984}, 'last_point': {'latitude': 39.961226704363476, 'longitude': 116.44382601069988}},
        {'first_point': {'latitude': 39.97527058333899, 'longitude': 116.3566986461087}, 'last_point': {'latitude': 40.08405971016491, 'longitude': 116.48009332679227}},
    ]

    queries["count"] = [
        {'latitude': 39.77837970523844, 'longitude': 116.2204541824271, 'radius': 986},
        {'latitude': 39.91422729845727, 'longitude': 116.31200951137352, 'radius': 971},
        {'latitude': 40.14336258671234, 'longitude': 116.39617518081226, 'radius': 925},
        {'latitude': 40.1447106182845, 'longitude': 116.54563889188094, 'radius': 919},
        {'latitude': 39.935718777017236, 'longitude': 116.50671860971669, 'radius': 935},
        {'latitude': 40.01434778741932, 'longitude': 116.23318520056601, 'radius': 903},
        {'latitude': 40.03012239452031, 'longitude': 116.37107536877156, 'radius': 960},
        {'latitude': 40.036931298117906, 'longitude': 116.34154998335516, 'radius': 948},
        {'latitude': 40.107526082703835, 'longitude': 116.3968459952813, 'radius': 936},
        {'latitude': 40.11127014901671, 'longitude': 116.40198574745405, 'radius': 982},
        {'latitude': 39.7928162906905, 'longitude': 116.47473621247438, 'radius': 999},
        {'latitude': 40.17991398251353, 'longitude': 116.5767270177861, 'radius': 939},
        {'latitude': 39.926692344506336, 'longitude': 116.38213951337272, 'radius': 953},
        {'latitude': 39.96713328669574, 'longitude': 116.21094733984714, 'radius': 966},
        {'latitude': 39.87716218744068, 'longitude': 116.34361694387914, 'radius': 917},
    ]

    queries["knn"] = [
        {'k': 134, 'latitude': 40.06354938823691, 'longitude': 116.41541982222077, 'time_first': 1202097586, 'time_last': 1202188227},
        {'k': 75, 'latitude': 39.8729401089201, 'longitude': 116.29078723993044, 'time_first': 1202311514, 'time_last': 1202358388},
        {'k': 78, 'latitude': 40.03570053441224, 'longitude': 116.08436533168631, 'time_first': 1202131071, 'time_last': 1202242685},
        {'k': 92, 'latitude': 39.15497362814973, 'longitude': 116.45110903788068, 'time_first': 1202103975, 'time_last': 1202436900},
        {'k': 91, 'latitude': 39.86715215741967, 'longitude': 116.87231128143556, 'time_first': 1202136343, 'time_last': 1202379460},
        {'k': 76, 'latitude': 39.74329993273813, 'longitude': 116.63582942390056, 'time_first': 1202294451, 'time_last': 1202314215},
        {'k': 150, 'latitude': 39.9994110941452, 'longitude': 116.64763741579796, 'time_first': 1202283850, 'time_last': 1202521934},
        {'k': 60, 'latitude': 40.23329355592637, 'longitude': 116.6971781705841, 'time_first': 1202045165, 'time_last': 1202435150},
        {'k': 126, 'latitude': 40.20075369744938, 'longitude': 116.91407765071632, 'time_first': 1202330524, 'time_last': 1202392189},
        {'k': 110, 'latitude': 39.50301701956862, 'longitude': 116.52023984498923, 'time_first': 1202108113, 'time_last': 1202142488},
        {'k': 144, 'latitude': 40.19457886361247, 'longitude': 116.23685870106785, 'time_first': 1202227359, 'time_last': 1202481956},
        {'k': 53, 'latitude': 39.15798237917503, 'longitude': 116.32802458491129, 'time_first': 1202056506, 'time_last': 1202086176},
        {'k': 124, 'latitude': 39.78691596685841, 'longitude': 116.58456387392044, 'time_first': 1202284834, 'time_last': 1202412982},
        {'k': 65, 'latitude': 39.860709549673125, 'longitude': 116.58972353451391, 'time_first': 1202107596, 'time_last': 1202235955},
        {'k': 71, 'latitude': 39.53911728558667, 'longitude': 116.90108323728315, 'time_first': 1202512903, 'time_last': 1202564952},
    ]

    queries["window"] = [
        {'first_point': {'latitude': 39.97436691982683, 'longitude': 116.37865299794458}, 'last_point': {'latitude': 39.95354239254571, 'longitude': 116.23329351809537}, 't1': 1201964280, 't2': 1202355386},
        {'first_point': {'latitude': 40.07798271833596, 'longitude': 116.48564280816765}, 'last_point': {'latitude': 40.05814484595434, 'longitude': 116.30078001480571}, 't1': 1202176482, 't2': 1202191792},
        {'first_point': {'latitude': 40.121896877705545, 'longitude': 116.29610794082426}, 'last_point': {'latitude': 39.807639625565606, 'longitude': 116.24899858139923}, 't1': 1201971722, 't2': 1202126202},
        {'first_point': {'latitude': 39.72921701099196, 'longitude': 116.55882426526766}, 'last_point': {'latitude': 39.905831919941036, 'longitude': 116.28358153711491}, 't1': 1201959546, 't2': 1202110076},
        {'first_point': {'latitude': 39.80916474083902, 'longitude': 116.47220720636398}, 'last_point': {'latitude': 39.70173692679674, 'longitude': 116.37126378901452}, 't1': 1202366142, 't2': 1202408864},
        {'first_point': {'latitude': 40.17359527135894, 'longitude': 116.2735724605651}, 'last_point': {'latitude': 39.79131957114456, 'longitude': 116.26098436606141}, 't1': 1202026764, 't2': 1202062880},
        {'first_point': {'latitude': 39.83311486884726, 'longitude': 116.20602083680838}, 'last_point': {'latitude': 39.897653641092084, 'longitude': 116.43950254073032}, 't1': 1201987899, 't2': 1202402006},
        {'first_point': {'latitude': 40.14809971927475, 'longitude': 116.36793908951098}, 'last_point': {'latitude': 39.929462743228605, 'longitude': 116.42642644009139}, 't1': 1202072309, 't2': 1202307229},
        {'first_point': {'latitude': 40.14404906761415, 'longitude': 116.49604642863252}, 'last_point': {'latitude': 39.763659679758476, 'longitude': 116.58266358195269}, 't1': 1202174579, 't2': 1202551865},
        {'first_point': {'latitude': 40.1190201393392, 'longitude': 116.39518131117683}, 'last_point': {'latitude': 39.91083953427202, 'longitude': 116.56557801913537}, 't1': 1201991979, 't2': 1202311987},
        {'first_point': {'latitude': 40.00462227737521, 'longitude': 116.46652202755284}, 'last_point': {'latitude': 40.146953669208585, 'longitude': 116.23693016450798}, 't1': 1202130478, 't2': 1202234961},
        {'first_point': {'latitude': 40.12723601952, 'longitude': 116.28631505864269}, 'last_point': {'latitude': 39.99074747539259, 'longitude': 116.51543150649371}, 't1': 1202326619, 't2': 1202389166},
        {'first_point': {'latitude': 39.880922850677464, 'longitude': 116.48169190645429}, 'last_point': {'latitude': 40.152032479077654, 'longitude': 116.30994169488014}, 't1': 1202081598, 't2': 1202285642},
        {'first_point': {'latitude': 39.832411241259734, 'longitude': 116.419742444611}, 'last_point': {'latitude': 40.01067229850092, 'longitude': 116.45489068938873}, 't1': 1202384275, 't2': 1202396005},
        {'first_point': {'latitude': 39.98747887203777, 'longitude': 116.22671485870016}, 'last_point': {'latitude': 39.95295239664779, 'longitude': 116.3029134310086}, 't1': 1202045257, 't2': 1202566481},
    ]

    # queries["where"] = create_where_queries(from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19",
    #                                        times=amount_of_individual_queries)
    # queries["distance"] = create_distance_queries(from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19",
    #                                              times=amount_of_individual_queries)
    # queries["when"] = create_when_queries(from_lon=116.2, to_lon=116.6, from_lat=39.7, to_lat=40.2,
    #                                       times=amount_of_individual_queries)
    # queries["how_long"] = create_how_long_queries(from_lon=116.2, to_lon=116.6, from_lat=39.7, to_lat=40.2,
    #                                               times=amount_of_individual_queries)
    #queries["count"] = create_count_queries(from_lon=116.2, to_lon=116.6, from_lat=39.7, to_lat=40.2, from_radius=900,
    #                                        to_radius=1000, times=amount_of_individual_queries)
    #queries["knn"] = create_knn_queries(from_lon=116, to_lon=117, from_lat=39, to_lat=40.5,
    #                                    from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19", from_k=50,
    #                                    to_k=150, times=amount_of_individual_queries)
    #queries["window"] = create_window_queries(from_lon=116.2, to_lon=116.6, from_lat=39.7, to_lat=40.2,
    #                                          from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19",
    #                                          times=amount_of_individual_queries)

    save_to_file({
        "filename": "queries_for_evaluation",
        "version": version
    }, queries)


def create_where_queries(from_date, to_date, times):
    where_queries = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())

    for _ in range(times):
        random_seconds = random.randint(0, total_seconds)
        random_dt = from_dt + timedelta(seconds=random_seconds)
        where_queries.append(random_dt.strftime("%Y-%m-%d %H:%M:%S"))

    return where_queries


def create_distance_queries(from_date, to_date, times):
    distance_queries = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())

    for _ in range(times):
        rand_sec_1 = random.randint(0, total_seconds)
        rand_sec_2 = random.randint(0, total_seconds)

        while rand_sec_1 == rand_sec_2:
            rand_sec_2 = random.randint(0, total_seconds)

        dt1 = from_dt + timedelta(seconds=rand_sec_1)
        dt2 = from_dt + timedelta(seconds=rand_sec_2)

        time_first = min(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        time_last = max(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")

        distance_queries.append({
            "time_first": time_first,
            "time_last": time_last
        })

    return distance_queries


def create_when_queries(from_lon, to_lon, from_lat, to_lat, times):
    when_queries = []
    for _ in range(times):
        when_queries.append(
            {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat)})
    return when_queries


def create_how_long_queries(from_lon, to_lon, from_lat, to_lat, times):
    how_long_queries = []
    for _ in range(times):
        how_long_queries.append({"first_point": {"longitude": random.uniform(from_lon, to_lon),
                                                 "latitude": random.uniform(from_lat, to_lat)},
                                 "last_point": {"longitude": random.uniform(from_lon, to_lon),
                                                "latitude": random.uniform(from_lat, to_lat)}})
    return how_long_queries


def create_count_queries(from_lon, to_lon, from_lat, to_lat, from_radius, to_radius, times):
    count_queries = []
    for _ in range(times):
        count_queries.append(
            {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat),
             "radius": random.randint(from_radius, to_radius)})
    return count_queries


def create_knn_queries(from_lon, to_lon, from_lat, to_lat, from_date, to_date, from_k, to_k, times):
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())
    knn_queries = []
    for _ in range(times):
        rand_sec_1 = random.randint(0, total_seconds)
        rand_sec_2 = random.randint(0, total_seconds)

        while rand_sec_1 == rand_sec_2:
            rand_sec_2 = random.randint(0, total_seconds)

        dt1 = from_dt + timedelta(seconds=rand_sec_1)
        dt2 = from_dt + timedelta(seconds=rand_sec_2)

        time_first = min(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        time_last = max(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")

        knn_queries.append({"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat),
                            "time_first": time_first, "time_last": time_last, "k": random.randint(from_k, to_k)})

    return knn_queries


def create_window_queries(from_lon, to_lon, from_lat, to_lat, from_date, to_date, times):
    window_queries = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())

    for _ in range(times):
        first_point = {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat)}
        last_point = {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat)}
        rand_sec_1 = random.randint(0, total_seconds)
        rand_sec_2 = random.randint(0, total_seconds)

        while rand_sec_1 == rand_sec_2:
            rand_sec_2 = random.randint(0, total_seconds)

        dt1 = from_dt + timedelta(seconds=rand_sec_1)
        dt2 = from_dt + timedelta(seconds=rand_sec_2)

        t1 = min(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        t2 = max(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        window_queries.append({"first_point": first_point, "last_point": last_point, "t1": t1, "t2": t2})

    return window_queries
