# OOD Benchmark for Ligand-Protein Affinity Prediction
Welcome to the OOD Testbench/Benchmark project designed for evaluating Out-of-Distribution (OOD) detection algorithms in the context of ligand-protein affinity prediction. This project serves as a comprehensive platform to assess the performance of various OOD detection techniques on a separate OOD dataset. It is the code source for the paper "Toward Out-of-Domain Binding Affinity Prediction"
## Algorithm Performance

Run on sbap_general_ec50_scaffold.json dataset.
In the presented results, each metric was calculated as the trimmed mean and standard deviation over 20 independent runs.

| Algos    | Val(ID)-ACC   | Val(ID)-AUC   | Val(OOD)-ACC  | Val(OOD)-AUC  | Test(ID)-ACC  | Test(ID)-AUC  | Test(OOD)-ACC | Test(OOD)-AUC |
|----------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| ERM      | 0.9647±0.0008 | 0.8805±0.0194 | 0.8990±0.0028 | 0.8397±0.0071 | 0.9218±0.0016 | 0.8494±0.0078 | 0.8399±0.0031 | 0.7900±0.0086 |
| MixUp    | 0.9632±0.0013 | 0.8464±0.0250 | 0.8946±0.0042 | 0.8230±0.0109 | 0.9200±0.0024 | 0.8317±0.0123 | 0.8326±0.0044 | 0.7665±0.0123 |
| DeepCoral| 0.9650±0.0015 | 0.8541±0.0176 | 0.9005±0.0021 | 0.8237±0.0140 | 0.9231±0.0020 | 0.8343±0.0121 | 0.8409±0.0022 | 0.7792±0.0128 |
| W2D      | 0.9648±0.0004 | 0.8708±0.0064 | 0.9016±0.0012 | 0.8403±0.0079 | 0.9230±0.0005 | 0.8454±0.0077 | 0.8412±0.0013 | 0.7821±0.0068 |
| PGD      | 0.9666±0.0012 | 0.8851±0.0124 | 0.9045±0.0011 | 0.8426±0.0093 | 0.9248±0.0011 | 0.8480±0.0087 | 0.8442±0.0013 | 0.7902±0.0076 |





### Environment Setup

1. **Install Dependencies**:
   To install all required packages, use the command:
   ```bash
   pip install -r requirement.txt
   ```
### Dataset download 

1. You can obtain the dataset for this project from the following link: [Download Dataset](https://drive.google.com/file/d/1fHmLCzGz57P-cJ7cVmKHfJajMyQKnCfs/view?usp=sharing)

    
### Algorithm Available

This project supports the following algorithms:

1. ERM (Empirical Risk Minimization)
2. IRM (Invariant Risk Minimization) 
3. MixUp
4. W2D (Two Dimesions of Worst-case Training)
5. PGD (Adversarial Attack)
6. DeepCoral
7. BIOW2D

### Running the project
```bash
cd BioFrontierOOD
python .\train.py --algorithm W2D --noise core --sort scaffold --value ec50 --model DeepDTA
```

1. The noise parameter can take values: core, refined, general
2. The sort parameter can take values: assay, scaffold, size
3. The value parameter can take values: ec50, ic50, ki, potency
4. The model parameter can take DeepDTA, GATNet, GCNNet, GINConvNet, GAT_GCN
