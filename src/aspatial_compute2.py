import pandas as pd
import numpy as np
from scipy import stats
import psycopg2
from sqlalchemy import *
from joblib import Parallel, delayed
import sys
import csv
import math
import argparse
import logging
from tqdm import tqdm
import yaml


log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
# Set a severity threshold
log.setLevel(logging.INFO)


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Launch egohoods computation (Jacobs)"
    )
    parser.add_argument('--aggregation', '-A',
                        default="upz",
                        choices=['upz', 'barrios', 'egohoods'])
    parser.add_argument('--parallelism', '-P',
                        default='20',
                        help='Number of parallel processes')
    return parser


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return average, math.sqrt(variance)


# calculate total size
def total_size(vector):
    return np.sum(vector)


def get_weights(vector):
    """ Calculate the vector weights
    :param vector: Positive vector
    :return: Vector weights
    :raise: TypeError if negative total size
    """
    ts = total_size(vector)
    if ts <= 0:
        raise TypeError('Vector must be positive')
    else:
        return np.true_divide(vector, ts)


def hhi(vector):
    """ Calculate the Hirschman-Herfindahl index
    :param vector: Positive vector
    :return: HHI (Float)
    """
    weights = get_weights(vector)
    n = weights.size
    if n == 0:
        return 0
    else:
        h = np.square(weights).sum()
    return 1 - h
    #return (h - 1.0 / n) / (1.0 - 1.0 / n)


def compute_jacobs_attributes(city, city_section, spatial_name, city_blocks, core_block):
    columns = [('a', 'b', 'c')]
    results_df = pd.DataFrame({}, index=pd.MultiIndex.from_tuples(columns, names=['sp_id', 'city', 'spatial_name']))
    # clean the dataframe
    results_df = results_df[results_df.index.get_level_values('sp_id') == 'Ras']

    conn = psycopg2.connect("dbname=crime-environment host=localhost user=denadai port=50013 password=lollone")
    conn.autocommit = True
    cur = conn.cursor()

    #log.info("City %s, section %s, name %s", city_section, city, spatial_name)

    #
    # Mixed land use
    #

    # LUM5_single
    cur.execute("select COALESCE(SUM(CASE WHEN l.use_type = %s THEN area END), 0) as res_sum, "
                "COALESCE(SUM(CASE WHEN l.use_type = %s THEN area END), 0) as non_res_sum, "
                "COALESCE(SUM(CASE WHEN l.use_type = %s THEN area END), 0) as recre_sum "
                "from land_uses l where l.bid = ANY(%s) and l.city = %s",
                ('residential', 'commercial', 'recreational', city_blocks, city))
    query_result = cur.fetchone()
    if query_result is None or np.sum(query_result) == 0:
        print("a")
        log.warning("land_use_mix3 %s %s %s", city_section, city, spatial_name)
        return results_df[results_df.index.get_level_values('sp_id') == 'Ras']
    entropy_stats = np.array(query_result)
    entropy = stats.entropy(entropy_stats)/np.log(len(entropy_stats))
    results_df.loc[(city_section, city, spatial_name), "land_use_mix3"] = entropy

    assert(0 <= entropy <= 1.0)

    cur.execute('select score '
                'from walk_index p '
                'where p.bid = %s AND p.city = %s',
                (core_block, city))
    query_result = cur.fetchone()
    if query_result is None:
        log.warning("walkscore %s %s %s", city_section, city, spatial_name)
        return results_df[results_df.index.get_level_values('sp_id') == 'Ras']
    results_df.loc[(city_section, city, spatial_name), 'core_walkscore'] = query_result[0]

    #
    # Small blocks
    #

    # avg_block_area
    cur.execute("select * from avg_block_area(%s::int[], %s)", (city_blocks, city))
    areas = []
    for r in cur.fetchall():
        areas.append(r[0])
    if len(areas) == 0:
        log.warning("avg_block_area %s %s %s", city_section, city, spatial_name)
        return results_df[results_df.index.get_level_values('sp_id') == 'Ras']
    avg_block_area = np.median(np.log(np.array(areas)+1))
    results_df.loc[(city_section, city, spatial_name), "avg_block_area"] = avg_block_area
    results_df.loc[(city_section, city, spatial_name), "n_blocks"] = len(areas)
    results_df.loc[(city_section, city, spatial_name), "sum_block_area"] = np.sum(np.log(np.array(areas)+1))

    #
    # Aged buildings and enterprises
    #

    # total area of the section
    cur.execute("select COALESCE(u.area, 0)::real, ST_Area(b.approx_geom::geography)/1000000::real, ST_Area(b.core_geom::geography)/1000000::real "
                "from spatial_groups b "
                "left join spatial_groups_unused_areas u on u.sp_id = b.sp_id and u.city = b.city "
                "AND b.spatial_name = u.spatial_name "
                "where b.sp_id = %s and b.city = %s AND b.spatial_name = %s ",
                (city_section, city, spatial_name))
    fetched = cur.fetchone()
    total_area = fetched[1]
    total_area_without_parksrivers = total_area - fetched[0]
    total_core_area = fetched[2]

    #assert(total_area_without_parksrivers > 0)
    if total_area_without_parksrivers < 0:
        log.warning("area zero %s %s %s", city_section, city, spatial_name)
        total_area_without_parksrivers = 0

    results_df.loc[(city_section, city, spatial_name), "area_filtr"] = total_area_without_parksrivers
    results_df.loc[(city_section, city, spatial_name), "area_tot"] = total_area
    results_df.loc[(city_section, city, spatial_name), "core_area_tot"] = total_core_area

    #
    # Density
    #
    cur.execute('select COALESCE(SUM(population), 0), '
                'CASE WHEN SUM(inforce)::float > 0 THEN 1-(SUM(employed)/SUM(inforce)::float)::float ELSE 0 END, '
                'COALESCE(SUM(dwellings), 0) '
                'from census '
                'where bid = ANY(%s) and city = %s',
                (city_blocks, city))
    query_result = cur.fetchone()
    if query_result is None:
        log.warning("census %s %s %s", city_section, city, spatial_name)
        return results_df[results_df.index.get_level_values('sp_id') == 'Ras']

    population = query_result[0]
    dwellings = query_result[2]
    results_df.loc[(city_section, city, spatial_name), "population"] = population
    results_df.loc[(city_section, city, spatial_name), "dwellings"] = dwellings
    results_df.loc[(city_section, city, spatial_name), "unemployed"] = query_result[1]

    if total_area_without_parksrivers == 0:
        results_df.loc[(city_section, city, spatial_name), "density_population"] = 0
        results_df.loc[(city_section, city, spatial_name), "density_dwellings"] = 0
    else:
        results_df.loc[(city_section, city, spatial_name), "density_population"] = query_result[0]/total_area_without_parksrivers
        results_df.loc[(city_section, city, spatial_name), "density_dwellings"] = dwellings/total_area_without_parksrivers

    # bld_area
    if total_area_without_parksrivers == 0:
        results_df.loc[(city_section, city, spatial_name), "building_density"] = 0
        results_df.loc[(city_section, city, spatial_name), "density_diversity"] = 0
    else:
        cur.execute('select COALESCE(area, 0) FROM building where bid = ANY(%s)', (city_blocks, ))
        values = []
        for r in cur.fetchall():
            values.append(r[0])
        results_df.loc[(city_section, city, spatial_name), "building_density"] = np.sum(values)/total_area_without_parksrivers
        results_df.loc[(city_section, city, spatial_name), "density_diversity"] = np.std(values)

    # Vacant land
    cur.execute("select COALESCE(SUM(COALESCE(area, 0)/1000000), 0) as vac_sum "
                    "from land_uses l where l.bid = ANY(%s) and l.city = %s AND l.use_type = %s",
                    (city_blocks, city, 'vacant'))
    query_result = cur.fetchone()
    vacant_land = 0
    if query_result and total_area_without_parksrivers > 0:
        vacant_land = query_result[0]
        vacant_land = vacant_land/total_area_without_parksrivers
    results_df.loc[(city_section, city, spatial_name), "vacant_land"] = vacant_land

    #
    # Covariates
    #
    '''
    # Property values
    cur.execute('select COALESCE(area, 1), COALESCE(value,0) '
                'from property_value b '
                'INNER JOIN blocks_group_with_building bb ON b.bid = bb.bid AND b.city = bb.city '
                'where b.bid = ANY(%s) AND b.city = %s AND value IS NOT NULL AND value > 0',
                (city_blocks, city))
    weights = []
    values = []
    weighted_avg, weighted_std = 0, 0
    for r in cur.fetchall():
        weights.append(1) # r[0]+
        values.append(r[1])
    if weights:
        weighted_avg, weighted_std = weighted_avg_and_std(values, weights)

    results_df.loc[(city_section, city, spatial_name), "property_values_avg"] = weighted_avg
    results_df.loc[(city_section, city, spatial_name), "building_diversity"] = weighted_std
    '''
    ## Alternative
    cur.execute('select COALESCE(area, 1), age '
                'from property_age b '
                'where b.bid = ANY(%s) AND b.city = %s AND age > 0',
                (city_blocks, city))
    weights = []
    values = []
    weighted_avg, weighted_std = 0, 0
    for r in cur.fetchall():
        weights.append(1)  # r[0]+
        values.append(r[1])
    if weights:
        weighted_avg, weighted_std = weighted_avg_and_std(values, weights)

    results_df.loc[(city_section, city, spatial_name), "building_diversity2"] = weighted_std

    # Residential stability
    cur.execute('select CASE SUM(total) WHEN 0 THEN 1 ELSE SUM(stable)/SUM(total)::float END as stable, '
                'CASE SUM(total2) WHEN 0 THEN 1 ELSE SUM(owner)/SUM(total2)::float END as owners '
                'from residential_stability b '
                'where b.bid = ANY(%s) AND b.city = %s ',
                (city_blocks, city))
    r = cur.fetchone()
    results_df.loc[(city_section, city, spatial_name), "residential_stable"] = r[0]
    results_df.loc[(city_section, city, spatial_name), "residential_owners"] = r[1]

    # Ethnic diversity
    cur.execute("select COALESCE(SUM(race1), 0), COALESCE(SUM(race2), 0), COALESCE(SUM(race3),0), COALESCE(SUM(race4),0), COALESCE(SUM(race5),0), COALESCE(SUM(race6),0) "
                "from ethnic_diversity e where e.bid = ANY(%s)",
                (city_blocks,))
    query_result = cur.fetchone()
    entropy_stats = np.array(query_result)
    if query_result is None or np.sum(query_result) == 0:
        entropy_stats = 0
    else:
        entropy_stats = hhi(entropy_stats)
    results_df.loc[(city_section, city, spatial_name), "ethnic_diversity"] = entropy_stats

    # Poverty percent
    cur.execute('select CASE SUM(total) WHEN 0 THEN 0 ELSE SUM(poors)/SUM(total)::float END '
                'from poverty_index b '
                'where b.bid = ANY(%s) AND b.city = %s',
                (city_blocks, city))
    query_result = cur.fetchone()
    results_df.loc[(city_section, city, spatial_name), "poverty_index"] = query_result[0]

    #
    cur.execute('select COALESCE(SUM(population), 0) from census where bid = %s and city = %s', (core_block, city))
    query_result = cur.fetchone()
    core_population = query_result[0]
    results_df.loc[(city_section, city, spatial_name), "core_population"] = core_population

    cur.execute('select num_Otrips_in, num_Otrips_out, attract '
                'FROM spatial_groups_trips s '
                'where s.sp_id = %s AND s.city = %s AND s.spatial_name = %s',
                (city_section, city, spatial_name))
    nin, nout, attract = 0, 0, 0
    r = cur.fetchone()
    if r:
        nin = r[0]
        nout = r[1]
        attract = r[2]

    results_df.loc[(city_section, city, spatial_name), "nin"] = nin
    results_df.loc[(city_section, city, spatial_name), "nout"] = nout
    results_df.loc[(city_section, city, spatial_name), "attractiveness"] = attract

    cur.execute("SELECT AVG(ST_DistanceSphere(ext.geom, w.w_geom)) from building ext "
                "inner join join_building_ways w on w.bid = ext.id "
                "where ext.bid = %s and ext.city = %s", (core_block, city))
    dist = 0
    result = cur.fetchone()
    if result:
        dist = result[0]
    results_df.loc[(city_section, city, spatial_name), 'distance_bld_roads'] = dist

    # Ambient
    cur.execute('select num_people '
                'from ambient_population b '
                'where b.bid = %s and b.city=%s',
                (core_block, city))
    query_result = cur.fetchone()
    n_people = 0
    if query_result:
        n_people = query_result[0]
    results_df.loc[(city_section, city, spatial_name), 'core_ambient'] = n_people

    # Baseline
    # 'Schools', 'Travel', 'Outdoor', 'Professional', 'Residence', ,  'Event'
    for c in ['Shops', 'Food', 'NightLife', 'Entertainment']:
        cur.execute('select COUNT(DISTINCT id) '
                    'FROM venues m '
                    'INNER JOIN spatial_groups s ON ST_Contains(s.approx_geom, m.position) '
                    'WHERE m.type = \'foursquare\' and m.parent_cat = %s '
                    'AND s.sp_id = %s and s.spatial_name = %s AND s.city = %s',
                    (c, city_section, spatial_name, city))
        query_result = cur.fetchone()
        n_pois = 0
        if query_result:
            n_pois = query_result[0]
        results_df.loc[(city_section, city, spatial_name), 'base_{}'.format(c.lower())] = n_pois

    cur.execute('select COUNT(DISTINCT id) '
                    'FROM venues m '
                    'INNER JOIN spatial_groups s ON ST_Contains(s.approx_geom, m.position) '
                    'WHERE m.type = \'foursquare\' and m.parent_cat IN (\'Shops\', \'Food\', \'NightLife\', \'Entertainment\')'
                    'AND s.sp_id = %s and s.spatial_name = %s AND s.city = %s',
                    (city_section, spatial_name, city))
    query_result = cur.fetchone()
    n_pois = 0
    if query_result:
        n_pois = query_result[0]
    results_df.loc[(city_section, city, spatial_name), 'npois'] = n_pois

    # Baseline
    for c in ['Shops', 'Food', 'NightLife', 'Schools', 'Entertainment', 'Residence', 'Travel', 'Outdoor', 'Professional', 'Event']:
        cur.execute('select COUNT(DISTINCT id) '
                    'FROM venues m '
                    'INNER JOIN spatial_groups s ON ST_Contains(s.core_geom, m.position) '
                    'WHERE m.type = \'foursquare\' and m.parent_cat = %s '
                    'AND s.sp_id = %s and s.spatial_name = %s AND s.city = %s',
                    (c, city_section, spatial_name, city))
        query_result = cur.fetchone()
        n_pois = 0
        if query_result:
            n_pois = query_result[0]
        results_df.loc[(city_section, city, spatial_name), 'core_{}'.format(c.lower())] = n_pois

    cur.close()
    conn.close()

    return results_df


def main():
    cities = ['boston', 'boston1m', 'bogota', 'bogota1m', 'chicago', 'chicago1m', 'LA', 'LA1m']
    spatial_names = ['ego', 'ego', 'ego', 'ego', 'ego', 'ego', 'ego', 'ego']

    parser = make_argument_parser()
    args = parser.parse_args()
    log.info("PARAMETERS %s", args)
    spatial_aggregation = args.aggregation

    with open('../config/postgres.yaml') as f:
        engine_configs = yaml.load(f, Loader=yaml.FullLoader)

    conn = psycopg2.connect("dbname={dbname} host={host} user={username} port={port} password={password}".format(**engine_configs))
    engine = create_engine('postgresql://{username}:{password}@{host}:{port}/{dbname}'.format(**engine_configs))
    conn.autocommit = True
    cur = conn.cursor()
    blocks = {}
    cores = {}

    for c, s in zip(cities, spatial_names):
        cur.execute('select sp_id, city, lower_ids, core_id from spatial_groups WHERE spatial_name = %s '
                    'AND city = %s ORDER by RANDOM()', (s, c, ))
        for r in cur.fetchall():
            blocks[(r[0], r[1], s)] = r[2]
            cores[(r[0], r[1], s)] = r[3]

    columns = [('a', 'b', 'c')]
    results_df = pd.DataFrame({}, index=pd.MultiIndex.from_tuples(columns, names=['sp_id', 'city', 'name']))
    # clean the dataframe
    results_df = results_df[results_df.index.get_level_values('sp_id') == 'Ras']

    df_list = Parallel(n_jobs=int(args.parallelism))(delayed(compute_jacobs_attributes)(city, group_id, spatial_name, city_blocks, cores[(group_id, city, spatial_name)]) for (group_id, city, spatial_name), city_blocks in tqdm(blocks.items()))
    for df in df_list:
        if not df.empty:
            results_df = results_df.append(df)
    results_df.to_csv('../data/generated_files/aspatial_features.csv'.format(sp_aggregation=spatial_aggregation))

    # crime
    log.info("SAVING CRIME")
    ucr_categories = []
    cur.execute("select distinct ucr1, ucr_category from crime")
    for r in cur.fetchall():
        ucr_categories.append((r[0], r[1]))

    with open('../data/generated_files/crime.csv', "w") as ofile:
        writer = csv.writer(ofile, delimiter=',')
        for (group_id, city, spatial_name), core_block in cores.items():
            for ucr1, ucr_category in ucr_categories:
                cur.execute("select SUM(num) from crime c where sp_id = %s and c.city = %s AND ucr1 = %s AND "
                            "ucr_category = %s "
                            "GROUP BY ucr1, ucr_category",
                            (core_block, city, ucr1, ucr_category))
                fetched = cur.fetchone()
                n = 0
                if fetched:
                    n = fetched[0]
                writer.writerow([group_id, city, spatial_name, n, ucr1, ucr_category])

    sql_where = []
    for c, s in zip(cities, spatial_names):
        sql_where.append("(a.city = '{}' and a.spatial_name = '{}')".format(c, s))
    sql_where = ' OR '.join(sql_where)

    # Spatial Matrix - 0-1
    log.info("COMPUTING EGOhoods matrix")
    sql = """
        select a.sp_id::text as o_sp_id, b.sp_id::text as d_sp_id,  a.city, a.spatial_name 
        from spatial_groups a, spatial_groups b 
        where a.spatial_name = b.spatial_name AND a.city = b.city and a.sp_id != b.sp_id 
        and (a.core_id = ANY(b.lower_ids) OR st_touches(a.core_geom, b.core_geom) OR st_intersects(a.core_geom, b.core_geom))
        AND ({})
        """.format(sql_where)
    edges_df = pd.read_sql_query(sql, con=engine)
    edges_df.to_parquet('../data/generated_files/egohoods_intersects.parquet')

    log.info("COMPUTING distance matrix")
    sql = """
        select a.sp_id::text as o_sp_id, b.sp_id::text as d_sp_id, 
        ST_Distance(ST_Centroid(a.core_geom)::geography, ST_Centroid(b.core_geom)::geography) as w, 
        a.city, a.spatial_name 
        from spatial_groups a, spatial_groups b 
        where a.spatial_name = b.spatial_name AND a.city = b.city AND a.sp_id < b.sp_id 
        AND ST_Distance(ST_Centroid(a.core_geom)::geography, ST_Centroid(b.core_geom)::geography) > 0
        AND ({})
        """.format(sql_where)
    edges_df = pd.read_sql_query(sql, con=engine)
    edges_df.to_parquet('../data/generated_files/spatial_dmatrix.parquet')

    cur.close()


if __name__ == '__main__':
    main()
