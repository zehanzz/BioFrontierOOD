# OOD Benchmark for Ligand-Protein Affinity Prediction (GNN)
Since the graph model takes different setting during the algorithm implementation, if you want to run the graph model, run this part.
### Algorithm Available

This project supports the following algorithms:

1. ERM (Empirical Risk Minimization)
2. MixUp
3. DeepCoral
4. W2D (Two Dimesions of Worst-case Training)
5. BIOW2D

### Running the project
```bash
cd BioFrontierOOD\graph
python .\train_graph.py --algorithm W2D --noise core --sort scaffold --value ec50 --model GINConvNet
```

1. The noise parameter can take values: core, refined, general
2. The sort parameter can take values: assay, scaffold, size
3. The value parameter can take values: ec50, ic50, ki, potency
5. The model parameter can take values: GINConvNet, GATNet, GCNNet, GAT_GCN

### Some Notice
When we run the W2D algorithm, since the different structure of each model, please see the comment under different comput_mask_smiles function.
