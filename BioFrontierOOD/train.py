import argparse
import json
import torch.optim as optim
from model import GATNet, GINConvNet, GAT_GCN, GCNNet, DeepDTA
from biodataset import CustomerDatasetGraph, CustomerDataset, CustomerDatasetM
# from graph_algorithms import ERM, DeepCoral, BIOW2D, MixUp, W2D
from algorithms import ERM, DeepCoral, MixUp, W2D, BIOW2D, IRM, PGD

def main(args):

    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    # Example usage
    config_path = "config.json"  # Path to your config file
    config = load_config(config_path)

    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    irm_lambda = config['irm_lambda']
    epsilons = config['epsilons']
    epsilonp = config['epsilonp']
    alphas = config['alphas']
    alphap = config['alphap']
    num_iter = config['num_iter']
    early_patience = config['early_patience']
    learn_patience = config['learn_patience']
    factor = config['factor']
    sample_ratio = config['sample_ratio']
    feature_ratio = config['feature_ratio']
    percentage = config['percentage']
    weight = config['weight']

    dataset = CustomerDatasetM(noise_type=args.noise, value_type=args.value, sort_type=args.sort)

    if args.model == "GATNet":
        model = GATNet()
    elif args.model == "GINConvNet":
        model = GINConvNet()
    elif args.model == "GCNNet":
        model = GCNNet()
    elif args.model == "GAT_GCN":
        model = GAT_GCN()
    elif args.model == "DeepDTA":
        model = DeepDTA(dataset.vocab_size_smiles, dataset.vocab_size_proteins)
    else:
        raise ValueError(f"Invalid model choice: {args.model}")
    # Instantiate optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if args.algorithm == "ERM":
        algorithm = ERM(model, dataset, num_epochs, optimizer)
    elif args.algorithm == "W2D":
        algorithm = W2D(model, dataset, num_epochs, optimizer, sample_ratio, feature_ratio, percentage, early_patience, learn_patience, factor)
    elif args.algorithm == "MixUp":
        algorithm = MixUp(model, dataset, num_epochs, optimizer)
    elif args.algorithm == "IRM":
        algorithm = IRM(model, dataset, num_epochs, optimizer, irm_lambda)
    elif args.algorithm == "PGD":
        algorithm = PGD(model, dataset, num_epochs, optimizer,epsilons, epsilonp, alphas, alphap, num_iter, early_patience, learn_patience, factor)
    elif args.algorithm == "BIOW2D":
        algorithm = BIOW2D(model, dataset, num_epochs, optimizer, sample_ratio, feature_ratio, percentage, early_patience, learn_patience, factor, weight)
    elif args.algorithm == "DeepCoral":
        algorithm = DeepCoral(model, dataset, num_epochs, optimizer, weight)
    else:
        raise ValueError("Invalid algorithm choice. Choose other algorithms.")

    algorithm.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate algorithms")
    parser.add_argument("--algorithm", choices=["ERM", "W2D", "MixUp", "IRM", "PGD", "DeepCoral", "BIOW2D"], required=True, help="Choose algorithm")
    parser.add_argument("--noise", choices=["core", "refined", "general"], default="core", help="Choose noise level")
    parser.add_argument("--sort", choices=["assay", "scaffold", "size", "protein", "protein_family"], default="scaffold", help="Choose sort type")
    parser.add_argument("--value", choices=["ec50", "ic50", "ki", "potency"], default="ic50", help="Choose value type")
    parser.add_argument("--model", choices=["GATNet", "GINConvNet", "GAT_GCN", "DeepDTA", "GCNNet"], required=True, help="Choose model")
    args = parser.parse_args()

    main(args)
