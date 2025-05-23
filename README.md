# MDS-MUD-DrugDrugInteraction

ML and NN pipelines for detecting interactions between drugs in text.

## Installation

1. Clone the repository:

```sh
git clone git@github.com:AimbotParce/MDS-MUD-DrugDrugInteraction.git
cd MDS-MUD-DrugDrugInteraction
```

2. Install the required packages:

```sh
pip install -r requirements.txt
```

> [!TIP]
> Use a virtual environment to avoid package conflicts.

3. Download the Stanford CoreNLP server, by following the instructions [here](https://stanfordnlp.github.io/CoreNLP/download.html).

> [!NOTE]
> Alternatively, you can use the Docker image for CoreNLP `nlpbox/corenlp`. Make sure to have Docker installed and run the following command:
> ```sh
> docker pull nlpbox/corenlp
> docker run -p 9000:9000 nlpbox/corenlp
> ```

## Usage 

1. Before running the script, make sure to have the CoreNLP server running:

```sh
cd ml
chmod +x corenlp-server.sh
./corenlp-server.sh
```

2. Then, run the ML-DDI script:

```sh 
chmod +x run.sh 
./run.sh
```

3. To execute the DL-DDI script: (TODO)

```sh
cd dl
chmod +x run.sh 
./run.sh
```

## Baselines

The following metrics were obtained using the provided baseline scripts:

### ML-DDI

```
                   tp	  fp	  fn	#pred	#exp	P	R	F1
------------------------------------------------------------------------------
advise             62	 255	  79	 317	 141	19.6%	44.0%	27.1%
effect            105	 132	 207	 237	 312	44.3%	33.7%	38.3%
int                16	  16	  12	  32	  28	50.0%	57.1%	53.3%
mechanism          78	 237	 183	 315	 261	24.8%	29.9%	27.1%
------------------------------------------------------------------------------
M.avg            -	-	-	-	-	34.7%	41.2%	36.4%
------------------------------------------------------------------------------
m.avg             261	 640	 481	 901	 742	29.0%	35.2%	31.8%
m.avg(no class)   404	 497	 338	 901	 742	44.8%	54.4%	49.2%
```

### DL-DDI

```
                   tp     fp      fn    #pred   #exp    P       R       F1
------------------------------------------------------------------------------
advise             80     40      61     120     141    66.7%   56.7%   61.3%
effect            150     80     162     230     312    65.2%   48.1%   55.4%
int                14      5      14      19      28    73.7%   50.0%   59.6%
mechanism          88     73     173     161     261    54.7%   33.7%   41.7%
------------------------------------------------------------------------------
M.avg            -      -       -       -       -       65.1%   47.1%   56.2%
------------------------------------------------------------------------------
m.avg             332    198     410     530     742    62.6%   44.7%   52.2%
m.avg(no class)   360    170     382     530     742    67.9%   48.5%   56.6%
```