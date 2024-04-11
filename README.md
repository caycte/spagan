### Prerequisites

Create a new conda environment to run the code.

- PyTorch
- PyTorch-geometric: https://pytorch-geometric.readthedocs.io/en/latest/
- networkx: https://networkx.github.io/
(We replaced all the dependencies from graph-tool which was a nightmare to download)


### Run the code:
Activating the corresponding conda env if you use a new conda environment.

```
python train.py --mode=SPAGAN --dataset=cora
```

### Structure:

The data is reorganized by dataset, and there are also 2 folders that correspond to the processed data for ADSF and SPAGAN.
For ADSF, you need to create lots of files in advance with the RWR process and the [dataset]_dijkstra.pkl.

The training saves the train curves in results\dataframes and the corresponding plots in results/plots.

### Model

*models_spagat.py*: loads the global model, where we can choose between different layers: GAT, ADSF, or SPAGAN within *SpGraphAttentionLayer* found in the file *layers_spagat.py* using the *mode* argument. They can also be combined; in that case, a combined mode would need to be added.

*layers_spagat.py* : *SpGraphAttentionLayer*:
- gat_layer : GAT layer
- adsf_layer: ADSF layer
- pathat_layer: SPAGAN layer


### Performance

#### Cora 

- GAT Test set results: accuracy= 0.82
- SPAGAN Test set results: loss= 0.8199 accuracy= 0.8340
- ADSF Test set results: loss= 0.6842 accuracy= 0.8270


### Link to other repositories

- GAT: https://github.com/PetarV-/GAT
- SPAGAN: https://github.com/ihollywhy/SPAGAN
- ADSF: https://github.com/AvigdorZ/ADaptive-Structural-Fingerprint