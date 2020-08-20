# Socio-economic, built environment, and mobility conditions associated with crime: A study of multiple cities

This repository contains all the code required to reproduce the results presented in the following paper:

> M. De Nadai et. al. *[Socio-economic, built environment, and mobility conditions associated with crime: A study of multiple cities](https://www.nature.com/articles/s41598-020-70808-2)* - *Scientific Reports 10, 13871 (2020)*.

> **Abstract:** *Nowadays, 23% of the world population lives in multi-million cities. In these metropolises, criminal activity is much higher and violent than in either small cities or rural areas. Thus, understanding what factors influence urban crime in big cities is a pressing need. Seminal studies analyse crime records through historical panel data or analysis of historical patterns combined with ecological factor and exploratory mapping. More recently, machine learning methods have provided informed crime prediction over time. However, previous studies have focused on a single city at a time, considering only a limited number of factors (such as socio-economical characteristics) and often at large in a single city. Hence, our understanding of the factors influencing crime across cultures and cities is very limited. Here we propose a Bayesian model to explore how violent and property crimes are related not only to socio-economic factors but also to the built environmental (e.g. land use) and mobility characteristics of neighbourhoods. To that end, we analyse crime at small areas and integrate multiple open data sources with mobile phone traces to compare how the different factors correlate with crime in diverse cities, namely Boston, Bogotá, Los Angeles and Chicago. We find that the combined use of socio-economic conditions, mobility information and physical characteristics of the neighbourhood effectively explain the emergence of crime, and improve the performance of the traditional approaches. However, we show that the socio-ecological factors of neighbourhoods relate to crime very differently from one city to another. Thus there is clearly no “one fits all” model.*


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

All the files with `(core)` in the name are used to test the alternative models where the features are computed at the core.

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

* [Entire database](https://figshare.com/articles/dataset/Socio-economic_built_environment_and_mobility_conditions_associated_with_crime_A_study_of_multiple_cities/7217729)

These files will be uploaded in a permanent repository upon the acceptance of the paper.
To load the entire database you can run:

``` sh
createdb crime-environment
gzip -d 2020_08_03.sqldump.gz
pg_restore -d crime-environment -U username -C 2020_08_03.sqldump
```

### Aggregated
* [generated_files](https://figshare.com/articles/dataset/Socio-economic_built_environment_and_mobility_conditions_associated_with_crime_A_study_of_multiple_cities/7217729) and place the extracted files in `data/generated_files/`

### Raw
* [Boston](https://figshare.com/articles/dataset/Socio-economic_built_environment_and_mobility_conditions_associated_with_crime_A_study_of_multiple_cities/7217729) and place the extracted files in `data/`
* [Bogota](https://figshare.com/articles/dataset/Socio-economic_built_environment_and_mobility_conditions_associated_with_crime_A_study_of_multiple_cities/7217729) and place the extracted files in `data/`
* [Chicago](https://figshare.com/articles/dataset/Socio-economic_built_environment_and_mobility_conditions_associated_with_crime_A_study_of_multiple_cities/7217729) and place the extracted files in `data/`
* [Los Angeles](https://figshare.com/articles/dataset/Socio-economic_built_environment_and_mobility_conditions_associated_with_crime_A_study_of_multiple_cities/7217729) and place the extracted files in `data/`





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
 
## License
This code is licensed under the MIT license. 


## Citation
If you find this work useful for your research, please cite our paper:

```
@article{de2020socio,
  title={Socio-economic, built environment, and mobility conditions associated with crime: A study of multiple cities},
  author={De Nadai, Marco and Xu, Yanyan and Letouz{\'e}, Emmanuel and Gonz{\'a}lez, Marta C and Lepri, Bruno},
  journal={arXiv preprint arXiv:2004.05822},
  year={2020}
}

```

## Acknowledgements 
We thank Paolo Bosetti and Junpeng Lao for the helpful comments. We especially thank Andrés Clavijo for his support on the data, we all hope that this work could make Bogotá better. This work was supported by the Berkeley DeepDrive and the ITS Berkeley 2018-19 SB1 Research Grant (to M.C.G.); the French Development Agency and the World Bank (to M.D.N., B.L. and E.L.).