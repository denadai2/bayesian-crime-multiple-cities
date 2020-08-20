--
-- PostgreSQL database dump
--

-- Dumped from database version 11.4 (Ubuntu 11.4-1.pgdg18.10+1)
-- Dumped by pg_dump version 11.4 (Ubuntu 11.4-1.pgdg18.10+1)

-- Started on 2020-08-14 21:47:58 CEST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 2 (class 3079 OID 619311)
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- TOC entry 4644 (class 0 OID 0)
-- Dependencies: 2
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry, geography, and raster spatial types and functions';


--
-- TOC entry 1500 (class 1255 OID 2302845)
-- Name: avg_block_area(integer[], text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.avg_block_area(blocks_id integer[], city text) RETURNS TABLE(area real)
    LANGUAGE sql STABLE
    AS $_$ Select ST_Area(geom::geography)::real from block as m where m.sp_id = ANY($1) and m.city = $2 AND greater_1sm IS FALSE $_$;


SET default_tablespace = '';

SET default_with_oids = false;

--
-- TOC entry 263 (class 1259 OID 732668)
-- Name: ambient_population; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ambient_population (
    bid integer NOT NULL,
    num_people double precision,
    city text NOT NULL
);


--
-- TOC entry 236 (class 1259 OID 620890)
-- Name: block_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.block_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 237 (class 1259 OID 620892)
-- Name: block; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.block (
    id integer DEFAULT nextval('public.block_id_seq'::regclass) NOT NULL,
    sp_id integer,
    geom public.geometry(MultiPolygon,4326),
    city text,
    geog public.geography(MultiPolygon,4326),
    greater_1sm boolean
);


--
-- TOC entry 238 (class 1259 OID 620899)
-- Name: building_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.building_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 239 (class 1259 OID 620901)
-- Name: building; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.building (
    id integer DEFAULT nextval('public.building_id_seq'::regclass) NOT NULL,
    bid integer,
    geom public.geometry(MultiPolygon,4326),
    floors integer,
    height double precision,
    city text,
    area double precision
);


--
-- TOC entry 240 (class 1259 OID 620908)
-- Name: block_building; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.block_building AS
 SELECT x.geom,
    x.sp_id,
    x.city,
    x.building_id,
    x.block_id,
    x.building_area
   FROM ( SELECT dtable.geom,
            dtable.sp_id,
            dtable.city,
            dtable.building_id,
            dtable.block_id,
            dtable.building_area,
            row_number() OVER (PARTITION BY dtable.geom ORDER BY dtable.area DESC) AS r
           FROM ( SELECT b.geom,
                    b.id AS building_id,
                    d.id AS block_id,
                    d.sp_id,
                    d.city,
                    b.area AS building_area,
                    public.st_area(public.st_intersection(b.geom, d.geom)) AS area
                   FROM (public.building b
                     JOIN public.block d ON ((public.st_intersects(b.geom, d.geom) AND (b.city = d.city))))) dtable
          ORDER BY dtable.area) x
  WHERE (x.r = 1)
  WITH NO DATA;


--
-- TOC entry 268 (class 1259 OID 1321986)
-- Name: block_centroids; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.block_centroids AS
 SELECT b1.id,
    b1.sp_id AS bid,
    b1.city,
    (public.st_centroid(b1.geom))::public.geography AS centroid
   FROM public.block b1
  WITH NO DATA;


--
-- TOC entry 241 (class 1259 OID 620927)
-- Name: blocks_group; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.blocks_group (
    bid integer NOT NULL,
    original_id text,
    geom public.geometry(MultiPolygon,4326),
    city text NOT NULL
);


--
-- TOC entry 242 (class 1259 OID 620933)
-- Name: blocks_group_bid_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.blocks_group_bid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4645 (class 0 OID 0)
-- Dependencies: 242
-- Name: blocks_group_bid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.blocks_group_bid_seq OWNED BY public.blocks_group.bid;


--
-- TOC entry 267 (class 1259 OID 895756)
-- Name: blocks_group_with_building; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.blocks_group_with_building AS
 SELECT DISTINCT b.bid,
    b.city
   FROM (public.blocks_group b
     JOIN public.building bu ON (((b.bid = bu.bid) AND (b.city = bu.city))))
  WITH NO DATA;


--
-- TOC entry 243 (class 1259 OID 620935)
-- Name: boundary; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.boundary (
    city text NOT NULL,
    geom public.geometry(MultiPolygon,4326)
);


--
-- TOC entry 244 (class 1259 OID 620941)
-- Name: buildings_vacuum_index; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.buildings_vacuum_index (
    bid integer,
    sim double precision
);


--
-- TOC entry 245 (class 1259 OID 620944)
-- Name: census; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.census (
    bid integer NOT NULL,
    population integer,
    employed integer,
    inforce integer,
    city text,
    tot_survey integer,
    dwellings integer
);


--
-- TOC entry 246 (class 1259 OID 620950)
-- Name: crawler_venues; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.crawler_venues (
    id character varying(255) NOT NULL,
    "position" public.geometry(Point,4326),
    category_id character varying(250),
    cityname character varying(150),
    type character varying(150) NOT NULL,
    position_geog public.geography(Point,4326),
    sid integer NOT NULL
);


--
-- TOC entry 247 (class 1259 OID 620956)
-- Name: crawler_venues_sid_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.crawler_venues_sid_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4646 (class 0 OID 0)
-- Dependencies: 247
-- Name: crawler_venues_sid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.crawler_venues_sid_seq OWNED BY public.crawler_venues.sid;


--
-- TOC entry 248 (class 1259 OID 620958)
-- Name: crime; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.crime (
    sp_id integer NOT NULL,
    num integer,
    city character varying(200) NOT NULL,
    ucr_category character varying(200) NOT NULL,
    ucr1 character varying(200) NOT NULL
);


--
-- TOC entry 249 (class 1259 OID 620964)
-- Name: ethnic_diversity; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ethnic_diversity (
    bid integer NOT NULL,
    city text NOT NULL,
    race1 integer,
    race2 integer,
    race3 integer,
    race4 integer,
    race5 integer,
    race6 integer
);


--
-- TOC entry 250 (class 1259 OID 620983)
-- Name: land_uses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.land_uses (
    use_type text NOT NULL,
    bid integer NOT NULL,
    area double precision,
    city text
);


--
-- TOC entry 251 (class 1259 OID 621034)
-- Name: player_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.player_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 252 (class 1259 OID 621036)
-- Name: pois_limit; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pois_limit (
    catname text,
    num integer
);


--
-- TOC entry 253 (class 1259 OID 621042)
-- Name: venues; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.venues (
    id text NOT NULL,
    "position" public.geometry(Point,4326),
    category_id text NOT NULL,
    cityname text NOT NULL,
    type text NOT NULL,
    position_geog public.geography(Point,4326),
    parent_cat text,
    sid integer NOT NULL
);


--
-- TOC entry 269 (class 1259 OID 1326766)
-- Name: pois_requests; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.pois_requests AS
 SELECT a.parent_cat,
    a.id,
    a.bid,
    round((public.st_x((a.centroid)::public.geometry))::numeric, 6) AS lon,
    round((public.st_y((a.centroid)::public.geometry))::numeric, 6) AS lat,
    array_to_string(array_agg(concat(round((public.st_x(a."position"))::numeric, 6), ',', round((public.st_y(a."position"))::numeric, 6))), ';'::text) AS dests,
    array_agg((public.st_distance(a.centroid, (a."position")::public.geography, false))::integer) AS dists,
    a.city
   FROM (( SELECT v.parent_cat,
            v.bid,
            v.id,
            v.centroid,
            v."position",
            v.city,
            v.dist,
            row_number() OVER (PARTITION BY v.parent_cat, v.bid, v.id ORDER BY v.dist) AS r
           FROM ( SELECT b2.parent_cat,
                    b1.bid,
                    b1.id,
                    b1.centroid,
                    b2."position",
                    b1.city,
                    (public.st_distance(b1.centroid, (b2."position")::public.geography, false))::integer AS dist
                   FROM (public.block_centroids b1
                     JOIN public.venues b2 ON (public.st_dwithin(b1.centroid, b2.position_geog, (1600)::double precision, false)))
                  WHERE (b2.type = 'foursquare'::text)) v) a
     JOIN public.pois_limit l ON (((l.catname = a.parent_cat) AND (a.r <= (l.num * 2)))))
  GROUP BY a.parent_cat, a.bid, a.id, (round((public.st_x((a.centroid)::public.geometry))::numeric, 6)), (round((public.st_y((a.centroid)::public.geometry))::numeric, 6)), a.city
  WITH NO DATA;


--
-- TOC entry 254 (class 1259 OID 621056)
-- Name: poverty_index; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.poverty_index (
    bid integer NOT NULL,
    poors double precision,
    city text NOT NULL,
    total integer
);


--
-- TOC entry 265 (class 1259 OID 784266)
-- Name: property_age; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.property_age (
    id integer NOT NULL,
    bid integer NOT NULL,
    age integer NOT NULL,
    area double precision,
    city text NOT NULL
);


--
-- TOC entry 264 (class 1259 OID 784264)
-- Name: property_age_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.property_age_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4647 (class 0 OID 0)
-- Dependencies: 264
-- Name: property_age_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.property_age_id_seq OWNED BY public.property_age.id;


--
-- TOC entry 270 (class 1259 OID 1334363)
-- Name: residential_density; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.residential_density (
    bid integer,
    city text,
    area double precision
);


--
-- TOC entry 255 (class 1259 OID 621095)
-- Name: residential_stability; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.residential_stability (
    bid integer NOT NULL,
    city text NOT NULL,
    total integer,
    stable integer,
    owner integer,
    total2 integer
);


--
-- TOC entry 256 (class 1259 OID 621103)
-- Name: spatial_groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spatial_groups (
    sp_id integer DEFAULT nextval('public.player_id_seq'::regclass) NOT NULL,
    city text NOT NULL,
    lower_ids integer[],
    spatial_name text NOT NULL,
    approx_geom public.geometry(MultiPolygon,4326),
    core_geom public.geometry(MultiPolygon,4326),
    core_id integer
);


--
-- TOC entry 257 (class 1259 OID 621110)
-- Name: spatial_groups_net_area; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spatial_groups_net_area (
    sp_id integer NOT NULL,
    city text NOT NULL,
    spatial_name text NOT NULL,
    used_area double precision
);


--
-- TOC entry 262 (class 1259 OID 732658)
-- Name: spatial_groups_trips; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spatial_groups_trips (
    sp_id integer NOT NULL,
    city text NOT NULL,
    spatial_name text NOT NULL,
    num_otrips_in double precision,
    num_otrips_out double precision,
    attract double precision,
    entropy_in double precision,
    entropy_out double precision
);


--
-- TOC entry 266 (class 1259 OID 803579)
-- Name: spatial_groups_unused_areas; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.spatial_groups_unused_areas AS
 SELECT s.sp_id,
    ((public.st_area((s.approx_geom)::public.geography) / (1000000)::double precision) - b.used_area) AS area,
    s.city,
    s.spatial_name
   FROM (public.spatial_groups s
     JOIN public.spatial_groups_net_area b ON (((b.sp_id = s.sp_id) AND (b.city = s.city) AND (s.spatial_name = b.spatial_name))))
  WITH NO DATA;


--
-- TOC entry 258 (class 1259 OID 621141)
-- Name: unused_areas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.unused_areas (
    geom public.geometry(MultiPolygon,4326),
    type text,
    city text
);


--
-- TOC entry 259 (class 1259 OID 621153)
-- Name: venue_categories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.venue_categories (
    parent_cat character varying(250) NOT NULL,
    cat character varying(250) NOT NULL
);


--
-- TOC entry 260 (class 1259 OID 621156)
-- Name: venues_sid_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.venues_sid_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4648 (class 0 OID 0)
-- Dependencies: 260
-- Name: venues_sid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.venues_sid_seq OWNED BY public.venues.sid;


--
-- TOC entry 261 (class 1259 OID 621158)
-- Name: walk_index; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.walk_index (
    bid integer,
    score double precision,
    city text
);


--
-- TOC entry 4392 (class 2604 OID 621164)
-- Name: blocks_group bid; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.blocks_group ALTER COLUMN bid SET DEFAULT nextval('public.blocks_group_bid_seq'::regclass);


--
-- TOC entry 4393 (class 2604 OID 621165)
-- Name: crawler_venues sid; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.crawler_venues ALTER COLUMN sid SET DEFAULT nextval('public.crawler_venues_sid_seq'::regclass);


--
-- TOC entry 4396 (class 2604 OID 784269)
-- Name: property_age id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.property_age ALTER COLUMN id SET DEFAULT nextval('public.property_age_id_seq'::regclass);


--
-- TOC entry 4394 (class 2604 OID 621167)
-- Name: venues sid; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.venues ALTER COLUMN sid SET DEFAULT nextval('public.venues_sid_seq'::regclass);


--
-- TOC entry 4475 (class 2606 OID 732675)
-- Name: ambient_population ambient_population_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ambient_population
    ADD CONSTRAINT ambient_population_pkey PRIMARY KEY (bid, city);


--
-- TOC entry 4404 (class 2606 OID 625479)
-- Name: block block_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.block
    ADD CONSTRAINT block_pkey PRIMARY KEY (id);


--
-- TOC entry 4419 (class 2606 OID 625481)
-- Name: blocks_group blocks_group_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.blocks_group
    ADD CONSTRAINT blocks_group_pkey PRIMARY KEY (bid, city);


--
-- TOC entry 4422 (class 2606 OID 625483)
-- Name: boundary boundary_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.boundary
    ADD CONSTRAINT boundary_pkey PRIMARY KEY (city);


--
-- TOC entry 4410 (class 2606 OID 625485)
-- Name: building building_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.building
    ADD CONSTRAINT building_pkey PRIMARY KEY (id);


--
-- TOC entry 4427 (class 2606 OID 625489)
-- Name: census census_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.census
    ADD CONSTRAINT census_pkey PRIMARY KEY (bid);


--
-- TOC entry 4430 (class 2606 OID 625491)
-- Name: crime crime_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.crime
    ADD CONSTRAINT crime_pkey PRIMARY KEY (sp_id, city, ucr1, ucr_category);


--
-- TOC entry 4433 (class 2606 OID 625493)
-- Name: ethnic_diversity ethnic_diversity_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ethnic_diversity
    ADD CONSTRAINT ethnic_diversity_pkey PRIMARY KEY (bid, city);


--
-- TOC entry 4436 (class 2606 OID 625495)
-- Name: land_uses land_uses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.land_uses
    ADD CONSTRAINT land_uses_pkey PRIMARY KEY (use_type, bid);


--
-- TOC entry 4446 (class 2606 OID 625505)
-- Name: poverty_index poverty_index_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.poverty_index
    ADD CONSTRAINT poverty_index_pkey PRIMARY KEY (bid, city);


--
-- TOC entry 4479 (class 2606 OID 784274)
-- Name: property_age property_age_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.property_age
    ADD CONSTRAINT property_age_pkey PRIMARY KEY (id);


--
-- TOC entry 4450 (class 2606 OID 625507)
-- Name: residential_stability residential_stability_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.residential_stability
    ADD CONSTRAINT residential_stability_pkey PRIMARY KEY (bid, city);


--
-- TOC entry 4461 (class 2606 OID 625511)
-- Name: spatial_groups_net_area spatial_groups_net_area_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups_net_area
    ADD CONSTRAINT spatial_groups_net_area_pkey PRIMARY KEY (sp_id, city, spatial_name);


--
-- TOC entry 4457 (class 2606 OID 625513)
-- Name: spatial_groups spatial_groups_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups
    ADD CONSTRAINT spatial_groups_pkey PRIMARY KEY (sp_id, city, spatial_name);


--
-- TOC entry 4472 (class 2606 OID 732665)
-- Name: spatial_groups_trips spatial_groups_trips_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups_trips
    ADD CONSTRAINT spatial_groups_trips_pkey PRIMARY KEY (sp_id, city, spatial_name);


--
-- TOC entry 4467 (class 2606 OID 625515)
-- Name: venue_categories venue_categories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.venue_categories
    ADD CONSTRAINT venue_categories_pkey PRIMARY KEY (parent_cat, cat);


--
-- TOC entry 4441 (class 2606 OID 625517)
-- Name: venues venues_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.venues
    ADD CONSTRAINT venues_pkey PRIMARY KEY (id, category_id, type);


--
-- TOC entry 4411 (class 1259 OID 625518)
-- Name: block_building_block_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_building_block_id_idx ON public.block_building USING btree (block_id);


--
-- TOC entry 4412 (class 1259 OID 625519)
-- Name: block_building_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_building_geom_idx ON public.block_building USING gist (geom);


--
-- TOC entry 4413 (class 1259 OID 625520)
-- Name: block_building_sp_id_city_building_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_building_sp_id_city_building_id_idx ON public.block_building USING btree (sp_id, city, building_id);


--
-- TOC entry 4484 (class 1259 OID 1321994)
-- Name: block_centroids_bid_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_centroids_bid_idx ON public.block_centroids USING btree (bid);


--
-- TOC entry 4485 (class 1259 OID 1321995)
-- Name: block_centroids_centroid_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_centroids_centroid_idx ON public.block_centroids USING gist (centroid);


--
-- TOC entry 4486 (class 1259 OID 1321993)
-- Name: block_centroids_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_centroids_id_idx ON public.block_centroids USING btree (id);


--
-- TOC entry 4399 (class 1259 OID 1947866)
-- Name: block_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_city_idx ON public.block USING btree (city);


--
-- TOC entry 4400 (class 1259 OID 625524)
-- Name: block_geog_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_geog_idx ON public.block USING gist (geog);


--
-- TOC entry 4401 (class 1259 OID 625532)
-- Name: block_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_geom_idx ON public.block USING gist (geom);


--
-- TOC entry 4402 (class 1259 OID 625533)
-- Name: block_greater_1sm_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_greater_1sm_idx ON public.block USING btree (greater_1sm);


--
-- TOC entry 4405 (class 1259 OID 625534)
-- Name: block_sp_id_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX block_sp_id_city_idx ON public.block USING btree (sp_id, city);


--
-- TOC entry 4414 (class 1259 OID 625536)
-- Name: blocks_group_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX blocks_group_city_idx ON public.blocks_group USING btree (city);


--
-- TOC entry 4415 (class 1259 OID 2206313)
-- Name: blocks_group_city_original_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX blocks_group_city_original_id_idx ON public.blocks_group USING btree (city, original_id);


--
-- TOC entry 4416 (class 1259 OID 625537)
-- Name: blocks_group_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX blocks_group_geom_idx ON public.blocks_group USING gist (geom);


--
-- TOC entry 4417 (class 1259 OID 625538)
-- Name: blocks_group_original_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX blocks_group_original_id_idx ON public.blocks_group USING btree (original_id);


--
-- TOC entry 4483 (class 1259 OID 895774)
-- Name: blocks_group_with_building_bid_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX blocks_group_with_building_bid_city_idx ON public.blocks_group_with_building USING btree (bid, city);


--
-- TOC entry 4420 (class 1259 OID 625539)
-- Name: boundary_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX boundary_city_idx ON public.boundary USING btree (city);


--
-- TOC entry 4406 (class 1259 OID 625540)
-- Name: building_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX building_city_idx ON public.building USING btree (city);


--
-- TOC entry 4407 (class 1259 OID 625541)
-- Name: building_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX building_geom_idx ON public.building USING gist (geom);


--
-- TOC entry 4408 (class 1259 OID 625542)
-- Name: building_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX building_id_idx ON public.building USING btree (bid, city);


--
-- TOC entry 4424 (class 1259 OID 625543)
-- Name: buildings_vacuum_index_bid_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX buildings_vacuum_index_bid_idx ON public.buildings_vacuum_index USING btree (bid);


--
-- TOC entry 4425 (class 1259 OID 625544)
-- Name: census_inforce_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX census_inforce_idx ON public.census USING btree (inforce);


--
-- TOC entry 4428 (class 1259 OID 625545)
-- Name: census_sp_id_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX census_sp_id_city_idx ON public.census USING btree (bid, city);


--
-- TOC entry 4431 (class 1259 OID 625546)
-- Name: ethnic_diversity_bid_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ethnic_diversity_bid_city_idx ON public.ethnic_diversity USING btree (bid, city);


--
-- TOC entry 4423 (class 1259 OID 625547)
-- Name: idx_boundary_geom; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_boundary_geom ON public.boundary USING gist (geom);


--
-- TOC entry 4434 (class 1259 OID 625559)
-- Name: land_uses_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX land_uses_city_idx ON public.land_uses USING btree (city);


--
-- TOC entry 4437 (class 1259 OID 625607)
-- Name: pois_limit_catname_num_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX pois_limit_catname_num_idx ON public.pois_limit USING btree (catname, num);


--
-- TOC entry 4487 (class 1259 OID 1326774)
-- Name: pois_requests_bid_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX pois_requests_bid_city_idx ON public.pois_requests USING btree (bid, city);


--
-- TOC entry 4488 (class 1259 OID 1326775)
-- Name: pois_requests_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX pois_requests_city_idx ON public.pois_requests USING btree (city);


--
-- TOC entry 4447 (class 1259 OID 625610)
-- Name: poverty_index_total_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX poverty_index_total_idx ON public.poverty_index USING btree (total);


--
-- TOC entry 4476 (class 1259 OID 2362751)
-- Name: property_age_bid_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX property_age_bid_city_idx ON public.property_age USING btree (bid, city);


--
-- TOC entry 4477 (class 1259 OID 2362750)
-- Name: property_age_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX property_age_city_idx ON public.property_age USING btree (city);


--
-- TOC entry 4448 (class 1259 OID 625616)
-- Name: residential_stability_bid_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX residential_stability_bid_city_idx ON public.residential_stability USING btree (bid, city);


--
-- TOC entry 4451 (class 1259 OID 625620)
-- Name: spatial_groups_approx_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_approx_geom_idx ON public.spatial_groups USING gist (approx_geom);


--
-- TOC entry 4452 (class 1259 OID 625621)
-- Name: spatial_groups_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_city_idx ON public.spatial_groups USING btree (city);


--
-- TOC entry 4453 (class 1259 OID 625622)
-- Name: spatial_groups_core_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_core_geom_idx ON public.spatial_groups USING gist (core_geom);


--
-- TOC entry 4454 (class 1259 OID 625623)
-- Name: spatial_groups_core_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_core_id_idx ON public.spatial_groups USING btree (core_id);


--
-- TOC entry 4455 (class 1259 OID 625624)
-- Name: spatial_groups_geography_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_geography_idx ON public.spatial_groups USING gist (public.geography(approx_geom));


--
-- TOC entry 4459 (class 1259 OID 625625)
-- Name: spatial_groups_net_area_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_net_area_city_idx ON public.spatial_groups_net_area USING btree (city);


--
-- TOC entry 4462 (class 1259 OID 625626)
-- Name: spatial_groups_net_area_spatial_name_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_net_area_spatial_name_city_idx ON public.spatial_groups_net_area USING btree (spatial_name, city);


--
-- TOC entry 4458 (class 1259 OID 625627)
-- Name: spatial_groups_spatial_name_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_spatial_name_city_idx ON public.spatial_groups USING btree (spatial_name, city);


--
-- TOC entry 4470 (class 1259 OID 732666)
-- Name: spatial_groups_trips_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_trips_city_idx ON public.spatial_groups_trips USING btree (city);


--
-- TOC entry 4473 (class 1259 OID 732667)
-- Name: spatial_groups_trips_spatial_name_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_trips_spatial_name_city_idx ON public.spatial_groups_trips USING btree (spatial_name, city);


--
-- TOC entry 4480 (class 1259 OID 803587)
-- Name: spatial_groups_unused_areas_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_unused_areas_city_idx ON public.spatial_groups_unused_areas USING btree (city);


--
-- TOC entry 4481 (class 1259 OID 803586)
-- Name: spatial_groups_unused_areas_sp_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_unused_areas_sp_id_idx ON public.spatial_groups_unused_areas USING btree (sp_id);


--
-- TOC entry 4482 (class 1259 OID 803588)
-- Name: spatial_groups_unused_areas_spatial_name_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX spatial_groups_unused_areas_spatial_name_idx ON public.spatial_groups_unused_areas USING btree (spatial_name);


--
-- TOC entry 4463 (class 1259 OID 806508)
-- Name: unused_areas_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX unused_areas_city_idx ON public.unused_areas USING btree (city);


--
-- TOC entry 4464 (class 1259 OID 625631)
-- Name: unused_areas_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX unused_areas_geom_idx ON public.unused_areas USING gist (geom);


--
-- TOC entry 4465 (class 1259 OID 806509)
-- Name: unused_areas_type_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX unused_areas_type_idx ON public.unused_areas USING btree (type);


--
-- TOC entry 4438 (class 1259 OID 625634)
-- Name: venues_cityname_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX venues_cityname_idx ON public.venues USING btree (cityname);


--
-- TOC entry 4439 (class 1259 OID 625635)
-- Name: venues_parent_cat_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX venues_parent_cat_idx ON public.venues USING btree (parent_cat);


--
-- TOC entry 4442 (class 1259 OID 625636)
-- Name: venues_position_geog_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX venues_position_geog_idx ON public.venues USING gist (position_geog);


--
-- TOC entry 4443 (class 1259 OID 625637)
-- Name: venues_position_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX venues_position_idx ON public.venues USING gist ("position");


--
-- TOC entry 4444 (class 1259 OID 625638)
-- Name: venues_type_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX venues_type_idx ON public.venues USING btree (type);


--
-- TOC entry 4468 (class 1259 OID 625639)
-- Name: walk_index_bid_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX walk_index_bid_idx ON public.walk_index USING btree (bid);


--
-- TOC entry 4469 (class 1259 OID 625640)
-- Name: walk_index_city_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX walk_index_city_idx ON public.walk_index USING btree (city);


--
-- TOC entry 4502 (class 2606 OID 625641)
-- Name: walk_index constraint_bid_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.walk_index
    ADD CONSTRAINT constraint_bid_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4493 (class 2606 OID 625646)
-- Name: crime constraint_bid_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.crime
    ADD CONSTRAINT constraint_bid_city FOREIGN KEY (sp_id, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4498 (class 2606 OID 625651)
-- Name: spatial_groups constraint_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups
    ADD CONSTRAINT constraint_city FOREIGN KEY (city) REFERENCES public.boundary(city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4491 (class 2606 OID 625656)
-- Name: blocks_group constraint_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.blocks_group
    ADD CONSTRAINT constraint_city FOREIGN KEY (city) REFERENCES public.boundary(city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4501 (class 2606 OID 625661)
-- Name: unused_areas constraint_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unused_areas
    ADD CONSTRAINT constraint_city FOREIGN KEY (city) REFERENCES public.boundary(city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4499 (class 2606 OID 625666)
-- Name: spatial_groups constraint_core_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups
    ADD CONSTRAINT constraint_core_id_city FOREIGN KEY (core_id, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4489 (class 2606 OID 625671)
-- Name: block constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.block
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (sp_id, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4492 (class 2606 OID 625676)
-- Name: census constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.census
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4497 (class 2606 OID 625681)
-- Name: residential_stability constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.residential_stability
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4494 (class 2606 OID 625686)
-- Name: ethnic_diversity constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ethnic_diversity
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4496 (class 2606 OID 625691)
-- Name: poverty_index constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.poverty_index
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4490 (class 2606 OID 625696)
-- Name: building constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.building
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4495 (class 2606 OID 625701)
-- Name: land_uses constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.land_uses
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4500 (class 2606 OID 625716)
-- Name: spatial_groups_net_area constraint_sp_id_city; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups_net_area
    ADD CONSTRAINT constraint_sp_id_city FOREIGN KEY (sp_id, city, spatial_name) REFERENCES public.spatial_groups(sp_id, city, spatial_name) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4504 (class 2606 OID 789939)
-- Name: ambient_population fk_ambient_population; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ambient_population
    ADD CONSTRAINT fk_ambient_population FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4505 (class 2606 OID 789927)
-- Name: property_age fk_property_age; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.property_age
    ADD CONSTRAINT fk_property_age FOREIGN KEY (bid, city) REFERENCES public.blocks_group(bid, city) ON UPDATE RESTRICT ON DELETE CASCADE;


--
-- TOC entry 4503 (class 2606 OID 789945)
-- Name: spatial_groups_trips fk_spatial_groups_trips; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spatial_groups_trips
    ADD CONSTRAINT fk_spatial_groups_trips FOREIGN KEY (sp_id, city, spatial_name) REFERENCES public.spatial_groups(sp_id, city, spatial_name) ON UPDATE RESTRICT ON DELETE CASCADE;


-- Completed on 2020-08-14 21:47:58 CEST

--
-- PostgreSQL database dump complete
--

