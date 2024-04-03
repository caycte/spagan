## SPAGAN in PyTorch

This is a PyTorch implementation of the paper "SPAGAN: Shortest Path Graph Attention Network"

### Prerequisites

We prefer to create a new conda environment to run the code.

#### PyTorch
Version >= 1.0.0

#### PyTorch-geometric
We use [torch_geometric](https://github.com/rusty1s/pytorch_geometric), [torch_scatter](https://rusty1s.github.io/pytorch_scatter/build/html/index.html) and [torch_sparse](https://github.com/rusty1s/pytorch_sparse) as backbone to implement the path attention mechanism. Please follow the [official website](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) to install them.

#### networkx 
We use [networkx](https://networkx.github.io/) to load the graph dataset.

#### graph-tool
We use graph-tool for fast APSP calculation. Please follow the [official website](https://graph-tool.skewed.de/) to install. 

### Run the code:
Activating the corresponding conda env if you use a new conda environment.

```
python train_spgat.py --mode=SPAGAN --dataset=cora
```

---
# Note Christine

## Model
*models_spagat.py*: load le modèle global, on peut choisir les différentes layers: GAT ou ADSF ou SPAGAN dans *SpGraphAttentionLayer* qui est dans le fichier *layers_spagat.py* avec l'argument *mode*. On peut les combiner aussi, il faut rajouter un mode combiné dans ce cas.

*layers_spagat.py* : *SpGraphAttentionLayer*:
- gat_layer : GAT layer
- adsf_layer: ADSF layer
- pathat_layer: SPAGAN layer


## Ce qui a été fait en plus du repo original de SPAGAN

- Ajout adsf_layer: ADSF layer dans *layers_spagat.py*
- Modification pipeline dans train_spagat.py en ajoutant l'argument adj_ad lorsqu'il le fallait et ajout de l'entraînement de ADSF en ajoutant le if mode == "ADSF"
- (dans utils.py pour load_data_orggcn.py j'ai essayé de merge les 2 façons de load les data entre le repo SPAGAN et ADSF pour avoir le adj_ad et ça tourne pour l'instant mais j'ai peut-être mal fait un truc parce que l'accuracy de SPAGAN est à 82% mtn alors qu'avant c'était à 84% donc à voir si on essaye de voir ce qui s'est mal passé ou si on modifie toute la pipeline pour avoir 2 façons de load les data entre SPAGAN ET ADSF) => ça marche finalement, pas besoin de changer
- supprimer les .cuda() car j'ai un mac
- fix les bug de SPAGAN originel pour que ça tourne

## Ce qu'il reste à faire

- comprendre un peu comment il entraine les 3 modèles les uns après les autres (python train_spagat.py à partir de la ligne 124)
- dans utils.py pour load_data_orggcn.py : corriger le load_data_orggcn.py pour ravoir les 84% d'accuracy du SPAGAN (voir repo originel et comparer avec le load_data de ADSF )
- peut-être ajouter un device en argument pour pouvoir faire tourner sur GPU et rajouter .to(device) là où j'ai supprimer .cuda()
- modifier le code pour pas que ce soit copier/coller
- évaluer les 3 modèles sur les différents dataset
- faire le rapport


## Performance pour le moment

Avec les paramètres par défaut juste en lançant:
python train_spagat.py

GAT Test set results: accuracy= 0.82
SPAGAN Test set results: loss= 0.8199 accuracy= 0.8340
ADSF Test set results: loss= 0.6842 accuracy= 0.8270
best result at epoch: 1853

## Note Max

### General
les données sont réorganisées par dataset, il y a aussi 2 folders qui correspondent aux données processées pour ADSF et SPAGAN. Je pense réorganiser l'ensemble pour avoir un data folder comme suit:

-data
    -processed
        -ADSF
        -SPAGAN
    -raw
        -citeseer
        -etc


l'entrainement enregistre les train curves dans results\dataframes et les plots correspondants dans results/plots. Je crois qu'il y a un problème dans la fonction qui trouve l'indice des images existantes (0.png, 1.png etc) car il me semble que pour le moment, si on run deux fois l'entrainement avec le meme model et dataset, les résultats sont overwrited.

Autre remarque: les train curves et val curves sont inversées sur les plots, pas compris d'ou venait le probleme.


### ADSF
il faut creer pleins de fichiers au préalable avec les RWR process et les [dataset]_dijkstra.pkl. Tout est dans le notebook adsf.ipynb
J'ai pas encore fait un entrainement complet d'ADSF

### SPAGAN
Ca marche sans graph tool, avec les meme perfs qu'avant


