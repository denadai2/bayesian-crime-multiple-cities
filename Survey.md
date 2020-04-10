

## Overall

| Paper | Multiple cities | Spatial auto-correlation | CI | Sources available | Public data
| ------ | :---: | :---: | :---: | :---: | :---: |
| [Bogomolov 2014](https://arxiv.org/pdf/1409.2983.pdf) |  |  |  |  |
| [Wang 2016](https://www.kdd.org/kdd2016/papers/files/adp1044-wangA.pdf) | | | | | x |
| [Graif 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2911240/) | | x | | | x |
| [Andersen 2005](https://academic.oup.com/bjc/article-abstract/46/2/258/355560?redirectedFrom=fulltext) | | x | | |  |
| [Hipp 2013](https://pdfs.semanticscholar.org/3a51/f1ad7f5aeb17960ce485f746460401acebe0.pdf) | x | x | | |  |
| [Mburu 2016](https://www.tandfonline.com/doi/abs/10.1080/24694452.2016.1163252?journalCode=raag21) | | x | x* | | x |
| [Malleson 2016](https://www.sciencedirect.com/science/article/pii/S0047235216300198) |  | ? |  |  |
| [Haining 2008](https://www.sciencedirect.com/science/article/pii/S0167947308003940) |  | x | | | x |
| [Andersen 2009](https://www.tandfonline.com/doi/abs/10.1080/00330124.2010.547151) |  | x | | | |

## Spatial aggregation

| Paper | Aggregation | Cities | 
| ------ | --- | --- |
| [Bogomolov 2014](https://arxiv.org/pdf/1409.2983.pdf) | grid | London | 
| [Wang 2016](https://www.kdd.org/kdd2016/papers/files/adp1044-wangA.pdf) | neighborhood | Chicago | 
| [Graif 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2911240/) | neighborhood | Chicago | 
| [Andersen 2005](https://academic.oup.com/bjc/article-abstract/46/2/258/355560?redirectedFrom=fulltext) | neighborhood | Vancouver |
| [Hipp 2013](https://pdfs.semanticscholar.org/3a51/f1ad7f5aeb17960ce485f746460401acebe0.pdf) | egohood, tracts | Buffalo, Chicago, Cincinnati, Cleveland, Dallas, Los Angeles, Sacramento, St. Louis, Tucson |
| [Mburu 2016](https://www.tandfonline.com/doi/abs/10.1080/24694452.2016.1163252?journalCode=raag21) | administrative units | London | 
| [Malleson 2016](https://www.sciencedirect.com/science/article/pii/S0047235216300198) | LSOA | London | 
| [Haining 2008](https://www.sciencedirect.com/science/article/pii/S0167947308003940) | Districts | Sheffield, UK |


## Methods

| Paper | Method | Spatial auto-correlation type |
| ------ | --- |  --- | 
| [Bogomolov 2014](https://arxiv.org/pdf/1409.2983.pdf) | Random Forest | - |
| [Wang 2016](https://www.kdd.org/kdd2016/papers/files/adp1044-wangA.pdf) | NB | - | 
| [Graif 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2911240/) | OLS | GWR | 
| [Andersen 2005](https://academic.oup.com/bjc/article-abstract/46/2/258/355560?redirectedFrom=fulltext) | OLS | SE |
| [Mburu 2016](https://www.tandfonline.com/doi/abs/10.1080/24694452.2016.1163252?journalCode=raag21) | NB | SE |
| [Malleson 2016](https://www.sciencedirect.com/science/article/pii/S0047235216300198) | Corr |  |
| [Haining 2008](https://www.sciencedirect.com/science/article/pii/S0167947308003940) | Poisson, NB | *CAR, SF |

