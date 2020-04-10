# Socio-economic, built environment, and mobility conditions associated with crime: A study of multiple cities

This repository contains all the code required to reproduce the results presented in the following paper:

* M. De Nadai et. al. *Socio-economic, built environment, and mobility conditions associated with crime: A study of multiple cities*, 2020 - *Submitted*.


# Dependencies

Dependencies are listed in the `requirements.txt` file at the root of the repository. Using [Python 3.6](https://www.python.org/downloads/) with [pip](https://pip.pypa.io/en/stable/installing/) all the required dependencies can be installed automatically.

``` sh
pip3 install -r install/requirements.txt
```

* [PostgreSQL 10.0](https://www.postgresql.org/) 
* [PostGIS 2.4.1](https://postgis.net) extension

# Code

The code of the analysis in divided in two parts: the Python scripts and modules used to support the analysis, and the notebooks where the outputs of the analysis have been produced.

## Scripts

### Preprocess

* `src/preprocess/bogota.ipynb` : script used for the pre-processing of all Bogota's (open) data.
* `src/preprocess/bogota-closed.ipynb` : script used for the pre-processing of all Bogota's (closed) data (e.g. mobile phone data).
* `src/preprocess/boston.ipynb` : script used for the pre-processing of all Boston's (open) data.
* `src/preprocess/boston-closed.ipynb` : script used for the pre-processing of all Boston's (closed) data (e.g. mobile phone data).
* `src/preprocess/LA.ipynb` : script used for the pre-processing of all Los Angeles' (open) data.
* `src/preprocess/LA-closed.ipynb` : script used for the pre-processing of all Los Angeles' (closed) data (e.g. mobile phone data).
* `src/preprocess/chicago.ipynb` : script used for the pre-processing of all Chicago's data.

### Computations
* `aspatial_compute.py` : script used to compute all the features from PostgreSQL.
* `prepare_features.ipynb` : script used to generate the consolidated dataset that is used in the regression.
* `experiments_all.bash` : Bash script lo launch all the experiments. There are some parameters and you can choose what experiments to launch.
* `experiments_ego.bash` : Bash script lo launch all the experiments about the corehood size. There are some parameters and you can choose what experiments to launch.
* `experiments_all.bash` : Bash script lo launch all the experiments about the spatial matrices. There are some parameters and you can choose what experiments to launch.
* `models_results.ipynb` : script to see the results of all the models.
* `manuscript_posteriorplots.ipynb` : script used to produce some of the images of the manuscript.
* `manuscript_maps.ipynb` : script used to produce some of the images of the manuscript.
* `manuscript_extra_plots.ipynb` : script used to create some supplementary plots.
* `pystan_fit4.py` : script used to do all the fits (non necessary if you use the aforementioned bash scripts).

To do the preprocess steps, please configure `postgres_example.yaml` and rename it to `postgres.yaml`.

# Data

### Computed
You can download the following files:

* [Entire database](https://drive.google.com/file/d/1AzrfcvcSc7ePNegPWTH8idIMqE-aUvQ9/view?usp=sharing)
* [Spatial neighbouring matrix](https://drive.google.com/file/d/1Lno2815esZHHuR_Sp1S38QPulmlDcaAJ/view?usp=sharing)
* [Spatial distance matrix](https://drive.google.com/file/d/1014Veo1QW3oPQg0gTemlhUYSRSSJu7t4/view?usp=sharing)

These files will be uploaded in a permanent repository upon the acceptance of the paper.
To load the entire database you can run:

``` sh
createdb crime-environment
pg_restore -d crime-environment -U username -C 2020_04_08.sqldump.gz
```

### Aggregated
Available through permanent links upon the acceptance of the paper

### Raw
Available through permanent links upon the acceptance of the paper


## License
This code is licensed under the MIT license. 


# DIY Instructions

Here we generate the entire database from ground. To do so, we have to create the minimal setup from this command:

``` sh
psql crime-environment < data/SQL/schema.sql
```

## Additional dependencies
* [osrm v5.20.0](http://project-osrm.org/)
* [osmconvert 0.8.10](https://wiki.openstreetmap.org/wiki/Osmconvert)

## City by city
All files have to be included in a directory with the city name (`data/[cityname]`). All the cities we used in 
our paper have a script placed in `src/preprocess/[cityname].ipynb`, where you can see how we process the data.
Here we add some specific instructions that might not be easy to understand for those pages.


## Walkability
A OpenStreetMap file for each city has to be downloaded (preferably from [here](https://wiki.openstreetmap.org/wiki/Planet.osm)), and placed in `data/[cityname]/OSM`. 
Our algorithm requires to launch a osrm server, where all the paths will be requested.

To do so, first merge all the city-specific OSM files into one file. Here I assume you use our files in the project directory:

``` sh 
osmconvert data/bogota/OSM/output.osm -o=bogota.o5m
osmconvert data/boston/OSM/output.osm -o=boston.o5m
osmconvert data/LA/OSM/los-angeles_california.osm -o=LA.o5m
osmconvert data/chicago/OSM/illinois-latest.osm.pbf -o=chicago.o5m
osmconvert bogota.o5m boston.o5m LA.o5m chicago.o5m -o=together.o5m
osmconvert together.o5m -o=together.osm.pbf
```

Then, we go for osrm. These commands convert and compress the data, and launch the server.

``` sh 
osrm-extract -p profiles/foot.lua data/OSM/together.osm.pbf
osrm-contract data/OSM/together.osrm
osrm-routed data/OSM/together.osrm
```


## Point of interests
You can insert the point of interests in the `venue` table, that has this schema

```
                         Table "public.venues"
    Column     |         Type          | Collation | Nullable | Default
---------------+-----------------------+-----------+----------+---------
 id            | text                  |           | not null |
 position      | geometry(Point,4326)  |           |          |
 category_id   | text                  |           |          |
 cityname      | text                  |           | not null |
 type          | text                  |           | not null |
 position_geog | geography(Point,4326) |           |          |
 parent_cat    | text                  |           |          |
 ```
 
 you can insert the position that is a PostGIS `Point`, the category (can be random), cityname, type (I used foursquare/OSM, you can use whethever you want), then parent_cat that is the category (Food, grocery etc).
 All other columns are optional. Then run this query:
 
 ```
 UPDATE venues SET position_geog=position::geography;
 ```
 
 Done!