from torch_geometric.data import InMemoryDataset, Data
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from utils import compute_metrics, print_result
from config import device

class ERM:
    def __init__(self, model, dataset, num_epochs, optimizer):
        self.model = model
        self.train_dataloader = dataset.dataloader_train
        self.iid_val_dataloader = dataset.dataloader_iid_val
        self.iid_test_dataloader = dataset.dataloader_iid_test
        self.ood_val_dataloader = dataset.dataloader_ood_val
        self.ood_test_dataloader = dataset.dataloader_ood_test
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.model.to(device)
        self.threshold = dataset.threshold
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.08036246056856665,
                                                        patience=3, verbose=True)
    def compute_metrics(self, output, cls_labels):
        return compute_metrics(output, cls_labels, self.threshold)

    def val_evaluate(self, dataloader, name):
        self.model.eval()
        val_running_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)
                num_proteins = len(data.proteins) // 1000
                data.proteins = data.proteins.reshape(num_proteins, 1000)
                output = self.model(data)
                loss = F.mse_loss(output, data.y.view(-1, 1).float().to(device))  # Calculate loss for this batch
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(data.cls_label.cpu().numpy())
                val_running_loss += loss.item()
            val_acc, auc = self.compute_metrics(all_predictions, all_labels)
            val_loss = val_running_loss / len(dataloader)
            return val_loss, val_acc

    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = 7
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)
                num_proteins = len(data.proteins) // 1000
                data.proteins = data.proteins.reshape(num_proteins, 1000)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.mse_loss(output, data.y.view(-1, 1).float().to(device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(data.cls_label.cpu().numpy())

            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader, 'IID_Val')
            print_result(epoch, train_loss, val_loss, val_acc)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                self.model.load_state_dict(best_model_state)
                break

    def evaluate(self, dataloader, name):
        self.model.eval()
        val_running_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)
                num_proteins = len(data.proteins) // 1000
                data.proteins = data.proteins.reshape(num_proteins, 1000)
                output = self.model(data)
                loss = F.mse_loss(output, data.y.view(-1, 1).float().to(device))
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(data.cls_label.cpu().numpy())
                val_running_loss += loss.item()

            output_array = np.array(all_predictions)
            binary = (output_array >= self.threshold).astype(int)
            acc = accuracy_score(binary, all_labels)
            val_loss = val_running_loss / len(dataloader)
            print(f'{name:30s} | Accuracy: {acc:.4f} | Loss: {val_loss:.4f}')
            return acc

    def run(self):
        self.train(self.train_dataloader)
        self.evaluate(self.iid_val_dataloader, "in-domain validation")
        self.evaluate(self.iid_test_dataloader, "in-domain test")
        self.evaluate(self.ood_val_dataloader, "out-of-domain validation")
        self.evaluate(self.ood_test_dataloader, "out-of-domain test")


class MixUp(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                        factor=0.08036246056856665,
                                                        patience=5, verbose=True)
    def mixup_data(self, data, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        lam_tensor = torch.tensor(lam, dtype=torch.float, device=data.x.device)

        mixed_data = data.clone()

        unique_graphs = data.batch.unique()
        for graph_id in unique_graphs:
            graph_mask = (data.batch == graph_id)
            num_nodes_in_graph = graph_mask.sum()

            if num_nodes_in_graph <= 0:
                continue

            perm = torch.randperm(num_nodes_in_graph.item(), device=data.x.device)

            node_indices = torch.where(graph_mask)[0]
            mixed_data.x[node_indices] = (lam_tensor * data.x[node_indices] +
                                          (1 - lam_tensor) * data.x[node_indices][perm])

        return mixed_data

    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = 7
        for epoch in range(self.num_epochs):
            self.model.train()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            for batch_idx, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(device)
                num_proteins = len(batch_data.proteins) // 1000
                batch_data.proteins = batch_data.proteins.reshape(num_proteins, 1000)
                # MixUp data
                mixed_batch_data = self.mixup_data(batch_data)

                # Forward pass with mixed data
                self.optimizer.zero_grad()
                output = self.model(mixed_batch_data)
                loss = F.mse_loss(output, mixed_batch_data.y.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(batch_data.cls_label.cpu().numpy())

            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.iid_val_dataloader, 'IID_Val')
            print_result(epoch, train_loss, val_loss, val_acc)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                self.model.load_state_dict(best_model_state)
                break

class DeepCoral(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, weight):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.weight = weight
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                        factor=0.08036246056856665,
                                                        patience=7, verbose=True)
    def coral_loss(self, source, target):
        d = source.data.shape[1]
        source_coral = (source - torch.mean(source, 0)).t() @ (source - torch.mean(source, 0))
        target_coral = (target - torch.mean(target, 0)).t() @ (target - torch.mean(target, 0))
        loss = torch.norm(source_coral - target_coral, p='fro') / (4 * d * d)
        return loss

    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = 14

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_coral_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []

            # Assume we are using the same dataloader structure as ERM for both source and target
            for batch_idx, (source_data, target_data) in enumerate(zip(self.train_dataloader, self.ood_val_dataloader)):
                source_data = source_data.to(device)
                target_data = target_data.to(device)

                # Reshaping if necessary, similar to what is done in ERM
                num_source_proteins = len(source_data.proteins) // 1000
                source_data.proteins = source_data.proteins.reshape(num_source_proteins, 1000)
                num_target_proteins = len(target_data.proteins) // 1000
                target_data.proteins = target_data.proteins.reshape(num_target_proteins, 1000)

                self.optimizer.zero_grad()

                source_output = self.model(source_data)
                target_output = self.model(target_data)

                mse_loss = F.mse_loss(source_output, source_data.y.view(-1, 1).float().to(device))
                coral_loss = self.coral_loss(source_output, target_output)
                # print(f"coral_loss: {coral_loss}, mse_loss: {mse_loss}")
                total_loss = mse_loss + self.weight * coral_loss

                total_loss.backward()
                self.optimizer.step()

                running_loss += mse_loss.item()
                running_coral_loss += coral_loss.item()

                all_predictions.extend(source_output.detach().cpu().numpy())
                all_labels.extend(source_data.cls_label.cpu().numpy())

            acc, _ = self.compute_metrics(all_predictions, all_labels)
            train_loss = running_loss / len(self.train_dataloader)
            train_coral_loss = running_coral_loss / len(self.train_dataloader)
            print(
                f'Epoch {epoch + 1}/{self.num_epochs}, Train Accuracy: {acc:.4f}, MSE Loss: {train_loss:.4f}, CORAL Loss: {train_coral_loss:.4f}')

            val_loss, val_acc = self.val_evaluate(self.iid_val_dataloader, 'IID_Val')
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                self.model.load_state_dict(best_model_state)
                break

        return best_val_loss


class BIOW2D(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, sample_ratio, feature_ratio, percentage, early_patience,
                 learn_patience, factor, weight):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.percentage = percentage
        self.early_patience = early_patience
        self.learn_patience = learn_patience
        self.factor = factor
        self.weight = weight
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.factor,
                                                        patience=self.learn_patience, verbose=True)

        # Load tokenizers
        self.tokenizer_proteins = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
        self.tokenizer_smiles = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')

    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = self.early_patience
        best_correct_smiles_proteins_tokens = []

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []

            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)

                num_proteins = len(data.proteins) // 1000
                data.proteins = data.proteins.reshape(num_proteins, 1000)

                proteins = data.proteins
                labels = data.y
                cls_labels = data.cls_label
                proteins, labels, cls_labels = proteins.to(device), labels.to(device), cls_labels.to(device)

                self.optimizer.zero_grad()
                output = self.model(data)

                if epoch / self.num_epochs < self.percentage:
                    loss = F.mse_loss(output, labels, reduction='none')
                    # Continue with the rest of your specific loss modification logic...
                else:
                    loss = F.mse_loss(output, labels)

                if epoch / self.num_epochs < self.percentage:
                    _, sorted_loss_index = torch.sort(loss, descending=True)
                    keep_size = int((1 - self.sample_ratio) * loss.size(0))
                    selected_indices = sorted_loss_index[:keep_size]

                    # Flatten the selected_indices tensor to 1D if necessary
                    if selected_indices.dim() > 1:
                        selected_indices = selected_indices.flatten()

                    data = self.subset_data(data, selected_indices)
                    proteins = data.proteins
                    cls_labels = data.cls_label
                    labels = data.y
                    output = self.model(data)
                    loss = F.mse_loss(output, labels)

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                grad_proteins = self.model.embedding_xt.weight.grad[proteins, :].to(device)
                topk_proteins = [torch.topk(grad_proteins[i].norm(dim=-1), int(self.feature_ratio * grad_proteins.shape[1]), largest=True)[1] for i in range(grad_proteins.shape[0])]
                proteins_mask_token_id = 0
                masked_proteins = proteins.clone()
                if topk_proteins:
                    for i in range(len(proteins)):
                        masked_proteins[i][topk_proteins[i]] = proteins_mask_token_id
                data.proteins = masked_proteins
                adv_output = self.model(data)
                loss = F.mse_loss(output, labels)
                adversarial_loss = F.kl_div(F.log_softmax(output, dim=0), F.softmax(adv_output, dim=0), reduction='batchmean')
                combined_loss = loss + self.weight * adversarial_loss
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())


            acc, _ = self.compute_metrics(all_predictions, all_labels)
            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.iid_val_dataloader, 'IID_Val')
            print_result(epoch, train_loss, val_loss, val_acc)

            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                self.model.load_state_dict(best_model_state)
                break

        return val_acc, best_correct_smiles_proteins_tokens

    def subset_data(self, data, selected_indices):
        # Ensure selected_indices is 1D
        if selected_indices.dim() > 1:
            selected_indices = selected_indices.view(-1)
        selected_indices = selected_indices.unique(sorted=True)  # Sort and remove duplicates

        new_data = Data()

        # Subset node features (x)
        new_data.x = data.x[selected_indices]

        # Subset edge index
        new_data.edge_index = self.filter_edges(data.edge_index, selected_indices)
        assert new_data.edge_index.size(1) > 0, "No edges left after subsetting."

        # Subset node or graph-level labels
        # if 'batch' in data:
        # print("HAve Batch!!")
        # This part assumes that you want to keep entire graphs that have any nodes in selected_indices
        unique_graphs = data.batch[selected_indices].unique(sorted=True)
        new_data.y = data.y[unique_graphs]
        new_data.cls_label = data.cls_label[unique_graphs]
        new_data.domain_id = data.domain_id[unique_graphs]
        new_data.proteins = data.proteins[unique_graphs]
        if 'c_size' in data:
            new_data.c_size = data.c_size[unique_graphs]

        # The number of unique graphs should match the number of graph-level attributes
        num_graphs = unique_graphs.size(0)
        assert new_data.y.size(0) == num_graphs, "Mismatch in graph labels after subsetting."
        assert new_data.cls_label.size(0) == num_graphs, "Mismatch in class labels after subsetting."
        assert new_data.domain_id.size(0) == num_graphs, "Mismatch in domain ids after subsetting."
        assert new_data.proteins.size(0) == num_graphs, "Mismatch in proteins after subsetting."
        if 'c_size' in data:
            assert new_data.c_size.size(0) == num_graphs, "Mismatch in c_size after subsetting."

        return new_data

    def filter_edges(self, edge_index, selected_indices):
        # Make sure selected_indices is a 1D tensor
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=edge_index.device)

        # Create a mask for the edges such that both the source and target nodes
        # are in the selected_indices
        mask = torch.isin(edge_index[0], selected_indices) & torch.isin(edge_index[1], selected_indices)

        # Apply mask to edge_index to filter edges
        new_edge_index = edge_index[:, mask]

        # Get the unique nodes in the filtered edge_index
        unique_nodes = new_edge_index.unique()

        # Map the old node indices to new ones based on the order in unique_nodes
        old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes.tolist())}

        # Remap the nodes in edge_index to their new values
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = old_to_new_indices[new_edge_index[0, i].item()]
            new_edge_index[1, i] = old_to_new_indices[new_edge_index[1, i].item()]

        return new_edge_index



class W2D(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, sample_ratio, feature_ratio, percentage, early_patience, learn_patience, factor):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.percentage = percentage
        self.train_dataloader = dataset.dataloader_train
        self.early_patience = early_patience
        self.learn_patience = learn_patience
        self.factor = factor
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.factor, patience=self.learn_patience, verbose=True)

    def subset_data(self, data, selected_indices):
        # Ensure selected_indices is 1D
        if selected_indices.dim() > 1:
            selected_indices = selected_indices.view(-1)
        selected_indices = selected_indices.unique(sorted=True)  # Sort and remove duplicates

        new_data = Data()

        # Subset node features (x)
        new_data.x = data.x[selected_indices]

        # Subset edge index
        new_data.edge_index = self.filter_edges(data.edge_index, selected_indices)
        assert new_data.edge_index.size(1) > 0, "No edges left after subsetting."

        # Subset node or graph-level labels
        # if 'batch' in data:
        # print("HAve Batch!!")
        # This part assumes that you want to keep entire graphs that have any nodes in selected_indices
        unique_graphs = data.batch[selected_indices].unique(sorted=True)
        new_data.y = data.y[unique_graphs]
        new_data.cls_label = data.cls_label[unique_graphs]
        new_data.domain_id = data.domain_id[unique_graphs]
        new_data.proteins = data.proteins[unique_graphs]
        if 'c_size' in data:
            new_data.c_size = data.c_size[unique_graphs]

        # The number of unique graphs should match the number of graph-level attributes
        num_graphs = unique_graphs.size(0)
        assert new_data.y.size(0) == num_graphs, "Mismatch in graph labels after subsetting."
        assert new_data.cls_label.size(0) == num_graphs, "Mismatch in class labels after subsetting."
        assert new_data.domain_id.size(0) == num_graphs, "Mismatch in domain ids after subsetting."
        assert new_data.proteins.size(0) == num_graphs, "Mismatch in proteins after subsetting."
        if 'c_size' in data:
            assert new_data.c_size.size(0) == num_graphs, "Mismatch in c_size after subsetting."

        return new_data
    def filter_edges(self, edge_index, selected_indices):
        # Make sure selected_indices is a 1D tensor
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=edge_index.device)

        # Create a mask for the edges such that both the source and target nodes
        # are in the selected_indices
        mask = torch.isin(edge_index[0], selected_indices) & torch.isin(edge_index[1], selected_indices)

        # Apply mask to edge_index to filter edges
        new_edge_index = edge_index[:, mask]

        # Get the unique nodes in the filtered edge_index
        unique_nodes = new_edge_index.unique()

        # Map the old node indices to new ones based on the order in unique_nodes
        old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes.tolist())}

        # Remap the nodes in edge_index to their new values
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = old_to_new_indices[new_edge_index[0, i].item()]
            new_edge_index[1, i] = old_to_new_indices[new_edge_index[1, i].item()]

        return new_edge_index

    def train(self, dataloader):
        best_val_loss = float('inf')  # Initialize the best OOD validation loss value
        counter = 0  # Counter to keep track of epochs without improvement
        patience = 3 # Number of epochs to wait before early stopping
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []
            for batch_idx, data in enumerate(self.train_dataloader):
                data = data.to(device)

                # Reshape proteins as done in ERM if required
                num_proteins = len(data.proteins) // 1000
                data.proteins = data.proteins.reshape(num_proteins, 1000)

                proteins = data.proteins
                labels = data.y
                cls_labels = data.cls_label
                proteins, labels, cls_labels = proteins.to(device), labels.to(device), cls_labels.to(device)

                self.optimizer.zero_grad()
                output = self.model(data)

                if epoch / self.num_epochs < self.percentage:
                    loss = F.mse_loss(output, labels, reduction='none')
                    # Continue with the rest of your specific loss modification logic...
                else:
                    loss = F.mse_loss(output, labels)

                if epoch / self.num_epochs < self.percentage:
                    _, sorted_loss_index = torch.sort(loss, descending=True)
                    keep_size = int((1 - self.sample_ratio) * loss.size(0))
                    selected_indices = sorted_loss_index[:keep_size]

                    # Flatten the selected_indices tensor to 1D if necessary
                    if selected_indices.dim() > 1:
                        selected_indices = selected_indices.flatten()

                    data = self.subset_data(data, selected_indices)
                    proteins = data.proteins
                    cls_labels = data.cls_label
                    labels = data.y
                    output = self.model(data)
                    loss = F.mse_loss(output, labels)

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                # Get the embeddings with get_features=True
                smiles_embeddings, proteins_embeddings = self.model(data, get_features=True)

                # Calculate gradients for embeddings
                # If GATNet, we can select the gcn1 layer, other is conv1
                grad_smiles = torch.autograd.grad(outputs=smiles_embeddings, inputs=list(self.model.conv1.parameters()),
                                                  grad_outputs=torch.ones_like(smiles_embeddings), retain_graph=True)[0]
                grad_proteins = torch.autograd.grad(outputs=proteins_embeddings, inputs=list(self.model.embedding_xt.parameters()),
                                    grad_outputs=torch.ones_like(proteins_embeddings), retain_graph=True)[0][proteins, :]
                # print(f"grad_embedding:{grad_proteins.shape}")
                # print(f"embedding:{proteins_embeddings.shape}")
                # Mask out the most important features (gradients)
                topk_smiles, mask_smiles = self.compute_mask_smiles(grad_smiles, smiles_embeddings, self.feature_ratio)
                mask_proteins = self.compute_mask(grad_proteins, proteins_embeddings, self.feature_ratio)

                # Apply masks to embeddings
                masked_smiles_embeddings = smiles_embeddings * mask_smiles
                masked_proteins_embeddings = proteins_embeddings * mask_proteins

                # Compute output using masked embeddings
                masked_output = self.model.forward_from_embeddings(masked_smiles_embeddings, masked_proteins_embeddings)
                masked_loss = F.mse_loss(masked_output, data.y.view(-1, 1).float().to(device))

                # Optimize based on masked output
                self.optimizer.zero_grad()
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += masked_loss.item()
                all_predictions.extend(masked_output.detach().cpu().numpy())
                all_labels.extend(data.cls_label.cpu().numpy())

            acc, auc = self.compute_metrics(all_predictions, all_labels)
            train_loss = running_loss / len(self.train_dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader, 'IID_Val')
            print_result(epoch, train_loss, val_loss, val_acc)

            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= self.early_patience:
                print("Early stopping triggered.")
                self.model.load_state_dict(best_model_state)
                break

        return val_acc

    def compute_mask_smiles(self, grad_embeddings, embeddings, feature_ratio):
        # GINConvNet
        num_features_to_mask = max(int(feature_ratio * embeddings.size(1)), 1)

        # Initialize the mask
        mask = torch.ones_like(embeddings).to(device)

        # Find top-k gradients and apply mask
        topk_indices = torch.topk(grad_embeddings.norm(dim=1), num_features_to_mask, largest=True).indices
        mask.scatter_(1, topk_indices.unsqueeze(0), 0)

        return topk_indices, mask



    # def compute_mask_smiles(self, grad_embeddings, embeddings, feature_ratio):
    #     # GAT
    #     # print(f"grad_embedding:{grad_embeddings.shape}")
    #     # print(f"embedding:{embeddings.shape}")
    #
    #     # Sum the gradients over the last dimension of grad_embeddings
    #     grad_embeddings_reduced = grad_embeddings.sum(dim=2)
    #
    #     # Make sure num_features_to_mask does not exceed the number of features in grad_embeddings_reduced
    #     num_features_to_mask = min(max(int(feature_ratio * embeddings.size(1)), 1), grad_embeddings_reduced.size(-1))
    #
    #     # Initialize the mask
    #     mask = torch.ones_like(embeddings).to(device)  # Assuming self.device is defined
    #
    #     # Find top-k gradients. Since grad_embeddings_reduced is 2D, we don't need to squeeze it.
    #     topk_values, topk_indices = torch.topk(grad_embeddings_reduced, num_features_to_mask, largest=True)
    #
    #     # Ensure topk_indices is broadcastable to the size of the mask. We expand it along the batch dimension.
    #     topk_indices = topk_indices.expand(mask.size(0), -1)
    #
    #     # Scatter zeros into the mask at the top k indices
    #     mask.scatter_(1, topk_indices, 0)
    #
    #     return topk_indices, mask

    # def compute_mask_smiles(self, grad_embeddings, embeddings, feature_ratio):
    #     # GAT_GCN
    #     # print(f"grad_embedding:{grad_embeddings.shape}")
    #     # print(f"embedding:{embeddings.shape}")
    #
    #     grad_embeddings_expanded = grad_embeddings.view(1, -1).expand(embeddings.size(0), -1)
    #
    #     # Calculate the number of features to mask based on the feature_ratio
    #     num_features_to_mask = min(max(int(feature_ratio * embeddings.size(1)), 1), embeddings.size(1))
    #
    #     # Initialize the mask
    #     mask = torch.ones_like(embeddings).to(device)  # Assuming 'device' is defined
    #
    #     # Find top-k gradients across the expanded grad_embeddings
    #     topk_values, topk_indices = torch.topk(grad_embeddings_expanded, num_features_to_mask, dim=1, largest=True)
    #
    #     # Scatter zeros into the mask at the top k indices
    #     mask.scatter_(1, topk_indices, 0)
    #
    #     return topk_indices, mask

    def compute_mask(self, grad_embeddings, embeddings, feature_ratio):
        batch_size, seq_length, num_features = embeddings.size()
        num_features_to_mask = int(feature_ratio * num_features)

        # Ensure we do not exceed the number of features
        num_features_to_mask = min(num_features_to_mask, num_features - 1)

        # Initialize the mask with ones
        mask = torch.ones_like(embeddings).to(embeddings.device)

        # Compute the norms of the gradients along the features dimension
        grad_norms = grad_embeddings.norm(p=2, dim=-1)

        for i in range(batch_size):
            for j in range(seq_length):
                # Make sure we're not asking for more top features than exist
                valid_k = min(num_features_to_mask, grad_norms[i, j].numel())

                # Find the top-k features with the largest gradients for each position in the sequence
                # Only proceed if there's at least one feature to mask
                if valid_k > 0:
                    top_values, topk_indices = grad_norms[i, j].topk(valid_k, largest=True)

                    # Only apply the mask if there are non-zero gradients
                    if top_values.sum() > 0:
                        # Set the mask to zero at the top-k positions
                        mask[i, j, topk_indices] = 0

        return mask



