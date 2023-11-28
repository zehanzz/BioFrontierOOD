import json
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import rbf_kernel
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from scipy import sparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader as GeometricDataLoader
# from torch_geometric.loader import DataLoader as GeometricDataLoader
import networkx as nx
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
import os
from torch_geometric.data import InMemoryDataset, Data
import seaborn as sns
import matplotlib.pyplot as plt
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.zeros((2048,))

def protein_to_embedding(protein_seq):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    embedding = [protein_seq.count(aa) for aa in amino_acids]
    return np.array(embedding) / len(protein_seq)

class CustomerDataset:
    def __init__(self, noise_type="general", value_type="ec50", sort_type="size"):
        self.threshold = None
        self.data_splits = None
        self.fix_value = None
        self.upper_bound = None
        self.lower_bound = None
        self.noise_type = noise_type
        self.value_type = value_type
        self.sort_type = sort_type
        self.setup_tokenizers()
        self.load_data()
        self.preprocess_data()
        self.setup_domain_ids()

    def setup_tokenizers(self):
        # self.tokenizer_smiles = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')

        # self.tokenizer_smiles = AutoTokenizer.from_pretrained('DeepChem/SmilesTokenizer_PubChem_1M')
        self.tokenizer_proteins = AutoTokenizer.from_pretrained('Rostlab/prot_bert')
        self.tokenizer_smiles = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
        self.vocab_size_smiles = self.tokenizer_smiles.vocab_size + 1
        self.vocab_size_proteins = self.tokenizer_proteins.vocab_size + 1
        # print("token ok")

    def get_data_file_path(self):
        return f"../drugood_all/sbap_{self.noise_type}_{self.value_type}_{self.sort_type}.json"

    def load_data(self):
        with open(self.get_data_file_path(), 'r') as file:
            data = json.load(file)
        self.setup_threshold(data)
        self.data_splits = self.setup_data_splits(data)
        # print("data ok")

    def setup_threshold(self, data):
        all_values = [case['reg_label'] for split in data['split'].values() for case in split]
        sorted_all_values = sorted(all_values)
        median = sorted_all_values[len(sorted_all_values) // 2]
        cfg = data['cfg']['classification_threshold']
        self.lower_bound, self.upper_bound, self.fix_value = cfg.values()
        self.threshold = median if self.lower_bound <= median <= self.upper_bound else self.fix_value

    def setup_data_splits(self, data):
        return {split: data['split'][split] for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']}

    def preprocess_data(self):
        for split, entries in self.data_splits.items():
            smiles_tensor, proteins_tensor= self.process_sequences(split)
            labels_array = np.array([entry['reg_label'] for entry in entries], dtype=np.float32)
            cls_labels_array = np.array([entry['cls_label'] for entry in entries], dtype=np.int32)
            domain_ids_array = np.array([entry['domain_id'] for entry in entries], dtype=np.int32)

            labels_tensor = torch.tensor(labels_array)
            cls_labels_tensor = torch.tensor(cls_labels_array)
            domain_ids_tensor = torch.tensor(domain_ids_array)

            dataset = TensorDataset(smiles_tensor, proteins_tensor, labels_tensor, cls_labels_tensor,
                                    domain_ids_tensor)
            setattr(self, f'dataset_{split}', dataset)
            # print(f'dataset_{split} created')  # Add this line
            loader = DataLoader(dataset, batch_size=32, shuffle=(split == 'train'))
            setattr(self, f'dataloader_{split}', loader)
            # print(f'dataloader_{split} created')  # Add this line


    def process_sequences(self, split):

        smiles_sequences = self.encode_sequences([entry['smiles'] for entry in self.data_splits[split]],
                                                 self.tokenizer_smiles)
        # Convert protein sequences to the expected format
        formatted_sequences = [' '.join(list(entry['protein'])) for entry in self.data_splits[split]]

        # Tokenize the formatted sequences
        proteins_sequences = self.encode_sequences(formatted_sequences, self.tokenizer_proteins)

        return torch.stack(smiles_sequences), torch.stack(proteins_sequences)

    def encode_sequences(self, sequences, tokenizer):
        max_length = 85 if tokenizer == self.tokenizer_smiles else 1200
        return [
            tokenizer.encode_plus(
                seq,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )['input_ids'].squeeze()
            for seq in sequences
        ]

    def setup_domain_ids(self):
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            setattr(self, f'domain_ids_{split}', [entry['domain_id'] for entry in self.data_splits[split]])




class CustomerDatasetM:
    def __init__(self, noise_type="general", value_type="ec50", sort_type="size"):
        self.threshold = None
        self.data_splits = None
        self.fix_value = None
        self.upper_bound = None
        self.lower_bound = None
        self.noise_type = noise_type
        self.value_type = value_type
        self.sort_type = sort_type
        self.load_data()
        self.setup_tokenizers()
        self.preprocess_data()
        self.setup_domain_ids()

    def report_dataset_details(self):
        print("Dataset Details:\n")
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            dataloader = getattr(self, f'dataloader_{split}')
            num_samples, class_distribution = self.calculate_class_distribution(dataloader)
            print(f"{split} Split: {num_samples} samples")
            print("Class Distribution:", class_distribution)
            print()
    def calculate_class_distribution(self, dataloader):
        class_counts = {}
        total_samples = 0
        for _, _, _, cls_labels_tensor, _ in dataloader:
            cls_labels = cls_labels_tensor.numpy()
            total_samples += len(cls_labels)
            for cls_label in cls_labels:
                class_counts[cls_label] = class_counts.get(cls_label, 0) + 1
        return total_samples, class_counts
    def get_class_distribution_data(self):
        distribution_data = {}
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            dataloader = getattr(self, f'dataloader_{split}')
            _, class_distribution = self.calculate_class_distribution(dataloader)
            distribution_data[split] = class_distribution
        return distribution_data

    def plot_class_distribution(self, distribution_data):
        fig, axs = plt.subplots(1, len(distribution_data), figsize=(15, 5), sharey=True)
        fig.suptitle('Class Distribution Across Different Dataset Splits')

        for i, (split, class_distribution) in enumerate(distribution_data.items()):
            sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()), ax=axs[i])
            axs[i].set_title(split)
            axs[i].set_xlabel('Class')
            axs[i].set_ylabel('Frequency' if i == 0 else '')

        plt.tight_layout()
        plt.show()
    def setup_tokenizers(self):
        self.tokenizer_smiles = Tokenizer(char_level=True, lower=False)
        self.tokenizer_proteins = Tokenizer(char_level=True, lower=False)

        # Fit Tokenizers on sequences
        smiles_sequences = [entry['smiles'] for split in self.data_splits.values() for entry in split]
        proteins_sequences = [entry['protein'] for split in self.data_splits.values() for entry in split]

        self.tokenizer_smiles.fit_on_texts(smiles_sequences)
        self.tokenizer_proteins.fit_on_texts(proteins_sequences)

        # Set vocab_size attributes
        self.vocab_size_smiles = len(self.tokenizer_smiles.word_index) + 1
        self.vocab_size_proteins = len(self.tokenizer_proteins.word_index) + 1
        # print("token ok")

    def get_data_file_path(self):
        return f"../drugood_all/sbap_{self.noise_type}_{self.value_type}_{self.sort_type}.json"

    def load_data(self):
        with open(self.get_data_file_path(), 'r') as file:
            data = json.load(file)
        self.setup_threshold(data)
        self.data_splits = self.setup_data_splits(data)
        # print("data ok")

    def setup_threshold(self, data):
        all_values = [case['reg_label'] for split in data['split'].values() for case in split]
        sorted_all_values = sorted(all_values)
        median = sorted_all_values[len(sorted_all_values) // 2]
        cfg = data['cfg']['classification_threshold']
        self.lower_bound, self.upper_bound, self.fix_value = cfg.values()
        self.threshold = median if self.lower_bound <= median <= self.upper_bound else self.fix_value

    def setup_data_splits(self, data):
        return {split: data['split'][split] for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']}

    def preprocess_data(self):
        for split, entries in self.data_splits.items():
            smiles_tensor, proteins_tensor = self.process_sequences(split)
            # print(proteins_tensor)
            labels_array = np.array([entry['reg_label'] for entry in entries], dtype=np.float32)
            cls_labels_array = np.array([entry['cls_label'] for entry in entries], dtype=np.int32)
            domain_ids_array = np.array([entry['domain_id'] for entry in entries], dtype=np.int32)

            labels_tensor = torch.tensor(labels_array)
            cls_labels_tensor = torch.tensor(cls_labels_array)
            domain_ids_tensor = torch.tensor(domain_ids_array)

            dataset = TensorDataset(smiles_tensor, proteins_tensor, labels_tensor, cls_labels_tensor,
                                    domain_ids_tensor)
            setattr(self, f'dataset_{split}', dataset)
            # print(f'dataset_{split} created')  # Add this line
            loader = DataLoader(dataset, batch_size=32, shuffle=(split == 'train'))
            setattr(self, f'dataloader_{split}', loader)
            # print(f'dataloader_{split} created')  # Add this line

    def process_sequences(self, split):
        smiles_sequences = self.encode_sequences([entry['smiles'] for entry in self.data_splits[split]],
                                                 self.tokenizer_smiles)
        proteins_sequences = self.encode_sequences([entry['protein'] for entry in self.data_splits[split]],
                                                   self.tokenizer_proteins)
        return torch.stack(smiles_sequences), torch.stack(proteins_sequences)

    def encode_sequences(self, sequences, tokenizer):
        # Encode Sequences and Pad them
        encoded_seqs = tokenizer.texts_to_sequences(sequences)
        max_length = 85 if tokenizer == self.tokenizer_smiles else 1200
        padded_seqs = pad_sequences(encoded_seqs, maxlen=max_length, padding='post')

        # Convert to torch tensors
        return [torch.tensor(seq) for seq in padded_seqs]

    def setup_domain_ids(self):
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            setattr(self, f'domain_ids_{split}', [entry['domain_id'] for entry in self.data_splits[split]])