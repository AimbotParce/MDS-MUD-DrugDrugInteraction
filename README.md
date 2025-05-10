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