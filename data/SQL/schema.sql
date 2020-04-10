--
-- Name: ambient_population; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE ambient_population (
    bid integer NOT NULL,
    num_people double precision,
    city text
);

ALTER TABLE ONLY ambient_population
    ADD CONSTRAINT ambient_population_pkey PRIMARY KEY (bid, city);

ALTER TABLE ambient_population
    ADD CONSTRAINT fk_ambient_population FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;

--
-- Name: block; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE block (
    id integer NOT NULL,
    sp_id integer,
    geom geometry(MultiPolygon,4326),
    city text,
    geog geography(MultiPolygon,4326),
    greater_1sm boolean
);

CREATE SEQUENCE block_id_seq;
ALTER TABLE ONLY block ALTER COLUMN id SET DEFAULT nextval('block_id_seq'::regclass);
ALTER TABLE ONLY block
    ADD CONSTRAINT block_pkey PRIMARY KEY (id);


CREATE INDEX block_geog_idx ON block USING gist (geog);
CREATE INDEX block_geom_idx ON block USING gist (geom);
CREATE INDEX block_greater_1sm_idx ON block USING btree (greater_1sm);
CREATE INDEX block_sp_id_city_idx ON block USING btree (sp_id, city);

ALTER TABLE block
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (sp_id, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: blocks_group; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE blocks_group (
    bid integer NOT NULL,
    original_id text,
    geom geometry(MultiPolygon,4326),
    city text
);


CREATE SEQUENCE blocks_group_bid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE blocks_group_bid_seq OWNED BY blocks_group.bid;
ALTER TABLE ONLY blocks_group ALTER COLUMN bid SET DEFAULT nextval('blocks_group_bid_seq'::regclass);
ALTER TABLE ONLY blocks_group
    ADD CONSTRAINT blocks_group_pkey PRIMARY KEY (bid, city);

CREATE INDEX blocks_group_city_idx ON blocks_group USING btree (city);
CREATE INDEX blocks_group_original_id_idx ON blocks_group USING btree (original_id);
CREATE INDEX blocks_group_geom_idx ON blocks_group USING gist (geom);

ALTER TABLE blocks_group
ADD CONSTRAINT constraint_city FOREIGN KEY (city) REFERENCES boundary (city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: boundary; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE boundary (
    city text,
    geom geometry(MultiPolygon,4326)
);

ALTER TABLE ONLY boundary
    ADD CONSTRAINT boundary_pkey PRIMARY KEY (city);
CREATE INDEX idx_boundary_geom ON boundary USING gist (geom);


--
-- Name: building; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE building (
    id integer NOT NULL,
    bid integer,
    geom geometry(MultiPolygon,4326),
    floors integer,
    height double precision,
    city text,
    area double precision
);

ALTER TABLE ONLY building
    ADD CONSTRAINT building_pkey PRIMARY KEY (id);

CREATE SEQUENCE building_id_seq;
ALTER TABLE ONLY building ALTER COLUMN id SET DEFAULT nextval('building_id_seq'::regclass);

CREATE INDEX building_id_idx ON building USING btree (bid, city);
CREATE INDEX building_geom_idx ON building USING gist (geom);
CREATE INDEX ON building (city);

ALTER TABLE building
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: median_income; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE median_income (
    lot_id text NOT NULL,
    bid integer NOT NULL,
    val double precision,
    city text
);

ALTER TABLE ONLY median_income
    ADD CONSTRAINT median_income_pkey PRIMARY KEY (lot_id, city);


--
-- Name: vacuum_index; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE vacuum_index (
    bid integer,
    score double precision
);

ALTER TABLE ONLY vacuum_index
    ADD CONSTRAINT vacuum_index_pkey PRIMARY KEY (bid);


--
-- Name: census; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE census (
    bid integer NOT NULL,
    population integer,
    employed integer,
    inforce integer,
    city text,
    tot_survey integer
);

ALTER TABLE ONLY census
    ADD CONSTRAINT census_pkey PRIMARY KEY (bid, city);

CREATE INDEX census_inforce_idx ON census USING btree (inforce);

ALTER TABLE census
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: crime; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE crime (
    sp_id integer NOT NULL,
    num integer,
    city text NOT NULL,
    ucr_category text NOT NULL,
    ucr1 text NOT NULL
);

ALTER TABLE ONLY crime
    ADD CONSTRAINT crime_pkey PRIMARY KEY (sp_id, city, ucr1, ucr_category);

ALTER TABLE crime
ADD CONSTRAINT constraint_bid_city FOREIGN KEY (sp_id, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: ethnic_diversity; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE ethnic_diversity (
    bid integer NOT NULL,
    city text NOT NULL,
    race1 integer,
    race2 integer,
    race3 integer,
    race4 integer,
    race5 integer,
    race6 integer
);

ALTER TABLE ONLY ethnic_diversity
    ADD CONSTRAINT ethnic_diversity_pkey PRIMARY KEY (bid, city);

CREATE INDEX ethnic_diversity_bid_city_idx ON ethnic_diversity USING btree (bid, city);

ALTER TABLE ethnic_diversity
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: land_uses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE land_uses (
    use_type text NOT NULL,
    bid integer NOT NULL,
    area double precision,
    city text
);


ALTER TABLE ONLY land_uses
    ADD CONSTRAINT land_uses_pkey PRIMARY KEY (use_type, bid);
CREATE INDEX land_uses_city_idx ON land_uses USING btree (city);

ALTER TABLE land_uses
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: poverty_index; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE poverty_index (
    bid integer NOT NULL,
    poors double precision,
    city text NOT NULL,
    total integer
);

ALTER TABLE ONLY poverty_index
    ADD CONSTRAINT poverty_index_pkey PRIMARY KEY (bid, city);

CREATE INDEX poverty_index_total_idx ON poverty_index USING btree (total);

ALTER TABLE poverty_index
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: property_value; Type: TABLE; Schema: public; Owner: -
--

create table property_value (bid integer, value float, area float);

CREATE INDEX ON property_value (bid);

ALTER TABLE property_value
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;
--
-- Name: property_age; Type: TABLE; Schema: public; Owner: -
--

create table property_age (
    id serial,
    bid integer NOT NULL,
    age integer NOT NULL,
    area float,
    city text NOT NULL
);

ALTER TABLE ONLY property_age
    ADD CONSTRAINT property_age_pkey PRIMARY KEY (id);
CREATE INDEX ON property_age (bid, city);
CREATE INDEX ON property_age (city);

ALTER TABLE property_age
    ADD CONSTRAINT fk_property_age FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- Name: spatial_groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE spatial_groups (
    sp_id integer NOT NULL,
    city text NOT NULL,
    lower_ids integer[],
    spatial_name text NOT NULL,
    approx_geom geometry(MultiPolygon,4326)
);

ALTER TABLE ONLY spatial_groups
    ADD CONSTRAINT spatial_groups_pkey PRIMARY KEY (sp_id, city, spatial_name);

CREATE INDEX spatial_groups_city_idx ON spatial_groups USING btree (city);
CREATE INDEX spatial_groups_spatial_name_city_idx ON spatial_groups USING btree (spatial_name, city);
CREATE INDEX spatial_groups_approx_geom_idx ON spatial_groups USING gist (approx_geom);
CREATE INDEX spatial_groups_geography_idx ON spatial_groups USING gist (geography(approx_geom));

ALTER TABLE spatial_groups
ADD CONSTRAINT constraint_core_id_city FOREIGN KEY (core_id, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;


CREATE TABLE spatial_groups_net_area (
    sp_id integer NOT NULL,
    city text NOT NULL,
    spatial_name text NOT NULL,
    used_area double precision
);
ALTER TABLE ONLY spatial_groups_net_area
    ADD CONSTRAINT spatial_groups_net_area_pkey PRIMARY KEY (sp_id, city, spatial_name);
CREATE INDEX spatial_groups_net_area_city_idx ON spatial_groups_net_area USING btree (city);
CREATE INDEX spatial_groups_net_area_spatial_name_city_idx ON spatial_groups_net_area USING btree (spatial_name, city);

ALTER TABLE spatial_groups_net_area
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (sp_id, city, spatial_name) REFERENCES spatial_groups  (sp_id, city, spatial_name) ON DELETE CASCADE ON UPDATE RESTRICT;



CREATE TABLE spatial_groups_trips (
    sp_id integer NOT NULL,
    city text NOT NULL,
    spatial_name text NOT NULL,
    num_Otrips_in double precision,
    num_Otrips_out double precision
);
ALTER TABLE ONLY spatial_groups_trips
    ADD CONSTRAINT spatial_groups_trips_pkey PRIMARY KEY (sp_id, city, spatial_name);
CREATE INDEX spatial_groups_trips_city_idx ON spatial_groups_trips USING btree (city);
CREATE INDEX spatial_groups_trips_spatial_name_city_idx ON spatial_groups_trips USING btree (spatial_name, city);

ALTER TABLE spatial_groups_trips
    ADD CONSTRAINT fk_spatial_groups_trips FOREIGN KEY (sp_id, city, spatial_name) REFERENCES spatial_groups (sp_id, city, spatial_name) ON UPDATE RESTRICT ON DELETE CASCADE;

--
-- Name: residential_stability; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE residential_stability (
    bid integer NOT NULL,
    city text NOT NULL,
    total integer,
    stable integer,
    owner integer,
    total2 integer
);

ALTER TABLE ONLY residential_stability
    ADD CONSTRAINT residential_stability_pkey PRIMARY KEY (bid, city);

CREATE INDEX residential_stability_bid_city_idx ON residential_stability USING btree (bid, city);

ALTER TABLE residential_stability
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES blocks_group (bid, city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: roads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE roads (
    id integer NOT NULL,
    geom geometry(MultiLineString,4326),
    city text
);

CREATE SEQUENCE roads_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE roads_id_seq OWNED BY roads.id;
ALTER TABLE ONLY roads ALTER COLUMN id SET DEFAULT nextval('roads_id_seq'::regclass);
ALTER TABLE ONLY roads
    ADD CONSTRAINT roads_pkey1 PRIMARY KEY (id);

CREATE INDEX roads_city_idx ON roads USING btree (city);
CREATE INDEX roads_geom_idx1 ON roads USING gist (geom);
CREATE INDEX roads_id_city_idx ON roads USING btree (id, city);

ALTER TABLE roads
ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (city) REFERENCES boundary (city) ON DELETE CASCADE ON UPDATE RESTRICT;

--
-- Name: venues; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE venues (
    id text NOT NULL,
    "position" geometry(Point,4326),
    category_id text,
    cityname text NOT NULL,
    type text NOT NULL,
    position_geog geography(Point,4326),
    parent_cat text
);

ALTER TABLE ONLY venues
    ADD CONSTRAINT pp PRIMARY KEY (id, type);

CREATE INDEX venues_cityname_idx ON venues USING btree (cityname);
CREATE INDEX venues_parent_cat_idx ON venues USING btree (parent_cat);
CREATE INDEX venues_type_idx ON venues USING btree (type);
CREATE INDEX venues_position_geog_idx ON venues USING gist (position_geog);
CREATE INDEX venues_position_idx ON venues USING gist ("position");


--
-- Name: unused_areas; Type: TABLE; Schema: public; Owner: nadai
--

CREATE TABLE unused_areas (
    geom geometry(MultiPolygon,4326),
    type text
);

CREATE INDEX unused_areas_geom_idx ON unused_areas USING gist (geom);
create index on unused_areas (city);
create index on unused_areas (type);

ALTER TABLE unused_areas
ADD CONSTRAINT constraint_city FOREIGN KEY (city) REFERENCES boundary (city) ON DELETE CASCADE ON UPDATE RESTRICT;

--- OTHERS

CREATE FUNCTION avg_block_area(blocks_id integer[], city text) RETURNS TABLE(area real) AS
$$ Select ST_Area(geom::geography)::real from block as m where m.sp_id = ANY($1) and m.city = $2 AND greater_1sm IS FALSE $$
STABLE
LANGUAGE SQL;


create materialized view join_building_ways as
select a.id as bid, w.id as way_id, w.geom as w_geom, a.area
from building a
cross join lateral (select id, geom from roads w where w.city=a.city order by a.geom <#> w.geom limit 1) w;
create index on join_building_ways (bid);
create index on join_building_ways (way_id);
create index on join_building_ways using gist (w_geom);


create materialized view block_w_buildings as
select cb.id
from block as cb
where exists (select id from building b where cb.sp_id = b.bid AND cb.city = b.city AND ST_Contains(cb.geom, b.geom));
create index on block_w_buildings (id);


create materialized view spatial_groups_unused_areas as
select s.sp_id, ST_Area(s.approx_geom::geography)/1000000 - used_area as area, s.city, s.spatial_name
from spatial_groups s
inner join spatial_groups_net_area b on b.sp_id = s.sp_id and b.city=s.city and s.spatial_name = b.spatial_name
;
create index on spatial_groups_unused_areas (sp_id);
create index on spatial_groups_unused_areas (city);
create index on spatial_groups_unused_areas (spatial_name);


CREATE MATERIALIZED VIEW block_centroids AS
select b1.id, b1.sp_id as bid, b1.city, ST_Centroid(b1.geom)::geography as centroid
FROM block b1;
CREATE INDEX ON block_centroids (id);
CREATE INDEX ON block_centroids (bid);
CREATE INDEX ON block_centroids using gist (centroid);


CREATE MATERIALIZED VIEW pois_requests AS
SELECT parent_cat, id, bid, round(ST_X(centroid::geometry)::numeric, 6) as lon, round(ST_Y(centroid::geometry)::numeric, 6) as lat, array_to_string(array_agg(concat(round(ST_X(position::geometry)::numeric, 6), ',', round(ST_Y(position::geometry)::numeric, 6))), ';') as dests, array_agg(ST_Distance(centroid, position, false)::int) as dists, city
FROM (
    SELECT v.*, ROW_NUMBER() OVER (PARTITION BY parent_cat, bid, id ORDER BY dist) AS r
    FROM (
        SELECT b2.parent_cat, b1.bid, b1.id, b1.centroid, b2.position, b1.city, ST_Distance(b1.centroid, b2.position, false)::int as dist
        FROM block_centroids b1
        INNER JOIN venues b2 ON ST_DWithin(b1.centroid, b2.position_geog, 1600, false)
        WHERE b2.type='foursquare'
    ) v
) a
INNER JOIN pois_limit l ON l.catname = a.parent_cat AND r <= l.num*2
GROUP BY parent_cat, bid, id, lon, lat, city;
CREATE INDEX ON pois_requests (bid, city);
CREATE INDEX ON pois_requests (city);


CREATE MATERIALIZED VIEW blocks_group_with_building AS
SELECT DISTINCT b.bid, b.city
FROM blocks_group b
INNER JOIN building bu ON b.bid = bu.bid AND b.city=bu.city;
CREATE INDEX ON blocks_group_with_building (bid, city);