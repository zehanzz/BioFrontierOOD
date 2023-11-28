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

def compute_kernels(smiles_data, protein_data, batch_size=2):
    num_batches = len(smiles_data) // batch_size
    kernel_drugs = []
    kernel_proteins = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        smiles_batch = smiles_data[start_idx:end_idx]
        protein_batch = protein_data[start_idx:end_idx]
        smiles_features = np.array([smiles_to_fingerprint(s) for s in smiles_batch])
        protein_features = np.array([protein_to_embedding(p) for p in protein_batch])
        kernel_drugs.append(rbf_kernel(smiles_features))
        kernel_proteins.append(rbf_kernel(protein_features))
    print(np.block(kernel_drugs).shape, np.block(kernel_proteins).shape)
    return np.block(kernel_drugs), np.block(kernel_proteins)



def compute_kronecker_kernel(matA, matB):
    # assert matA.shape[2] == matA.shape[3], "matA is not square"
    # assert matB.shape[2] == matB.shape[3], "matB is not square"
    matA = matA.astype(np.float32)
    matB = matB.astype(np.float32)
    return np.kron(matA, matB)
def kronRLS_train(K, Y, lambda_reg=0.1):
    np.fill_diagonal(K, np.diagonal(K) + lambda_reg)
    alpha = np.linalg.solve(K, Y)
    return alpha


def kronRLS_predict(K_test, alpha):
    return np.dot(K_test, alpha)
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

class CustomerDatasetGraph:
    def __init__(self, root='processed', noise_type="general", value_type="ec50", sort_type="size"):
        self.root = root
        self.threshold = None
        self.data_splits = None
        self.fix_value = None
        self.upper_bound = None
        self.lower_bound = None
        self.noise_type = noise_type
        self.value_type = value_type
        self.sort_type = sort_type
        # Check and load data, then set up tokenizers and domain ids.
        self.check_and_load_data()

    def check_and_load_data(self):
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            dataset_path = os.path.join(self.root, f"{split}_data.pt")
            if not os.path.isfile(dataset_path):
                print(f"Processing {split} data...")
                # Implement these methods according to your needs
                self.load_data()
                self.setup_tokenizers()
                self.setup_domain_ids()
                self.preprocess_data()
                break
            else:
                self.load_data()
                print(f"Data for {split} already processed, loading...")
                data, slices = torch.load(dataset_path)
                setattr(self, f'dataset_{split}', CustomInMemoryDataset(self.root, data, slices))

        self.create_dataloaders()

    def create_dataloaders(self):
        # This function will create dataloaders from the processed data.
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            dataset = getattr(self, f'dataset_{split}')
            loader = GeometricDataLoader(dataset, batch_size=32, shuffle=(split == 'train'), follow_batch=['x', 'proteins'])
            setattr(self, f'dataloader_{split}', loader)

    def setup_tokenizers(self):
        self.tokenizer_proteins = Tokenizer(char_level=True, lower=False)
        proteins_sequences = [entry['protein'] for split in self.data_splits.values() for entry in split]
        self.tokenizer_proteins.fit_on_texts(proteins_sequences)
        self.vocab_size_proteins = len(self.tokenizer_proteins.word_index) + 1

    def get_data_file_path(self):
        return f"../drugood_all/sbap_{self.noise_type}_{self.value_type}_{self.sort_type}.json"

    def load_data(self):
        with open(self.get_data_file_path(), 'r') as file:
            data = json.load(file)
        self.setup_threshold(data)
        self.data_splits = self.setup_data_splits(data)

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
        max_nodes, max_edges = self.get_max()
        for split, entries in self.data_splits.items():
            c_size, features, edge_index, proteins = self.process_sequences(split, max_nodes, max_edges)
            labels_array = np.array([entry['reg_label'] for entry in entries], dtype=np.float32)
            cls_labels_array = np.array([entry['cls_label'] for entry in entries], dtype=np.int32)
            domain_ids_array = np.array([entry['domain_id'] for entry in entries], dtype=np.int32)
            dataset = MyCustomDataset(c_size, features, edge_index, proteins, labels_array, cls_labels_array, domain_ids_array, split)
            loader = GeometricDataLoader(dataset, batch_size=32, shuffle=(split == 'train'))
            setattr(self, f'dataloader_{split}', loader)

    def process_sequences(self, split, max_nodes, max_edges):
        smiles_sequences = [entry['smiles'] for entry in self.data_splits[split]]
        proteins_sequences = self.encode_sequences([entry['protein'] for entry in self.data_splits[split]], self.tokenizer_proteins)
        # c_size, features, edge_index = [self.smile_to_graph(smiles, max_nodes, max_edges) for smiles in smiles_sequences]
        c_sizes = []
        features_list = []
        edge_indexes = []

        for smiles in smiles_sequences:
            c, features_single, edge_index_single = self.smile_to_graph(smiles, max_nodes, max_edges)
            c_sizes.append(c)
            features_list.append(features_single)
            edge_indexes.append(edge_index_single)
        # print([g[0].size() for g in graph_representations])
        return c_sizes, features_list, edge_indexes, proteins_sequences

    def get_max(self):
        max_nodes = 0
        max_edges = 0
        for split, entries in self.data_splits.items():
            smiles_sequences = [entry['smiles'] for entry in entries]
            for smiles in smiles_sequences:
                mol = Chem.MolFromSmiles(smiles)
                max_nodes = max(max_nodes, mol.GetNumAtoms())
                max_edges = max(max_edges, mol.GetNumBonds())
        print(max_nodes)
        print(max_edges)
        return max_nodes, max_edges
    def encode_sequences(self, sequences, tokenizer):
        encoded_seqs = tokenizer.texts_to_sequences(sequences)
        max_length = 1000
        padded_seqs = pad_sequences(encoded_seqs, maxlen=max_length, padding='post')
        return [torch.tensor(seq) for seq in padded_seqs]

    def setup_domain_ids(self):
        for split in ['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']:
            setattr(self, f'domain_ids_{split}', [entry['domain_id'] for entry in self.data_splits[split]])

    def smile_to_graph(self, smiles, max_nodes, max_edges):
        mol = Chem.MolFromSmiles(smiles)
        c_size = mol.GetNumAtoms()
        num_padding_nodes = max_nodes - c_size

        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))
        # Add padding nodes with a feature vector of zeros
        padding_features = [np.zeros_like(features[0]) for _ in range(num_padding_nodes)]
        features.extend(padding_features)

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        g = nx.Graph(edges).to_directed()
        edge_index = [[e1, e2] for e1, e2 in g.edges]

        # Calculate how many padding edges are needed.
        current_edge_count = len(edge_index)
        num_padding_edges = max_edges - current_edge_count

        # Add padding edges with a feature vector of zeros. Here, we choose a convention where a "dummy" edge
        # connects the last node to itself. This can be modified based on how you want to represent dummy edges.
        padding_edge_index = [[max_nodes - 1, max_nodes - 1] for _ in range(num_padding_edges)]  # max_nodes - 1 is the last valid node index
        edge_index.extend(padding_edge_index)

        return c_size, features, edge_index

    def atom_features(self, atom):
        return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class MyCustomDataset(InMemoryDataset):
    def __init__(self, c_size, features, edge_index, proteins, labels_array, cls_labels_array, domain_ids_array, split):
        # Initialising all the attributes for later use
        self.split = split
        self.c_size = c_size
        self.features = features
        self.edge_index = edge_index
        self.proteins = proteins
        self.labels_array = labels_array
        self.cls_labels_array = cls_labels_array
        self.domain_ids_array = domain_ids_array

        # Calling the parent constructor
        super(MyCustomDataset, self).__init__('.')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        # Create a list to store PyTorch Geometric Data objects
        data_list = []

        # Iterating over each sample and constructing a data object
        for i in range(len(self.labels_array)):
            data = Data(x=torch.Tensor(np.array(self.features[i])),
                        edge_index=torch.LongTensor(np.array(self.edge_index[i])).transpose(1, 0),
                        y=torch.FloatTensor([self.labels_array[i]]),  # Wrap the value in a list
                        cls_label=torch.IntTensor([self.cls_labels_array[i]]),  # Same for this one
                        domain_id=torch.IntTensor([self.domain_ids_array[i]]),  # and this one
                        proteins=torch.LongTensor(np.array(self.proteins[i])))

            # Adding c_size as an additional attribute to the data object
            data.__setitem__('c_size', torch.LongTensor([self.c_size[i]]))  # Here as well

            # Appending the data object to the data list
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CustomInMemoryDataset(InMemoryDataset):
    def __init__(self, root, data, slices, transform=None, pre_transform=None):
        super(CustomInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = data, slices

    @property
    def processed_file_names(self):
        return ['dummy.pt']  # This should return the names of files that are assumed to be in the processed_dir

    def process(self):
        pass  # No need to process since data is already processed
if __name__ == '__main__':

    dataset = CustomerDatasetGraph()
    train_loader = dataset.dataloader_train
    iid_val_loader = dataset.dataloader_iid_val
    iid_test_loader = dataset.dataloader_iid_test
    ood_val_loader = dataset.dataloader_ood_val
    ood_test_loader = dataset.dataloader_ood_test

    num_train_samples = len(train_loader.dataset)
    num_iid_val_samples = len(iid_val_loader.dataset)
    num_iid_test_samples = len(iid_test_loader.dataset)
    num_ood_val_samples = len(ood_val_loader.dataset)
    num_ood_test_samples = len(ood_test_loader.dataset)

    for batch_idx, data in enumerate(train_loader):
        # We will just check the first batch to test if custom_collate is working
        print(f"Batch {batch_idx + 1}")
        print(data)  # print the data to see the structure (you might want to print specific parts of the data)
        break  # break after the first batch

    print(f"Number of samples in train loader: {num_train_samples}")
    print(f"Number of samples in IID validation loader: {num_iid_val_samples}")
    print(f"Number of samples in IID test loader: {num_iid_test_samples}")
    print(f"Number of samples in OOD validation loader: {num_ood_val_samples}")
    print(f"Number of samples in OOD test loader: {num_ood_test_samples}")
