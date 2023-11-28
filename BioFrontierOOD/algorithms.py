import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import copy
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from torch.optim import lr_scheduler
import numpy as np
from utils import compute_metrics, transfer_to_device, print_result
from config import device
import warnings
warnings.filterwarnings("ignore")

class ERM:
    def __init__(self, model, dataset, num_epochs, optimizer):
        self.model = model
        self.model.to(device)
        # Get dataloader from dataset
        self.train_dataloader = dataset.dataloader_train
        self.iid_val_dataloader = dataset.dataloader_iid_val
        self.iid_test_dataloader = dataset.dataloader_iid_test
        self.ood_val_dataloader = dataset.dataloader_ood_val
        self.ood_test_dataloader = dataset.dataloader_ood_test
        # Set the training config
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.08036246056856665, patience=3, verbose=True)
        # Get the pre-defined threshold for accuracy calculation
        self.threshold = dataset.threshold

    def compute_metrics(self, output, cls_labels):
        return compute_metrics(output, cls_labels, self.threshold)

    def val_evaluate(self, dataloader):
        self.model.eval()
        val_running_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                output = self.model(smiles, proteins)
                output = output.squeeze(1)
                loss = F.mse_loss(output, labels)  # Calculate loss for this batch
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())
                val_running_loss += loss.item()
            val_acc, auc = self.compute_metrics(all_predictions, all_labels)
            val_loss = val_running_loss / len(dataloader)
            return val_loss, val_acc

    def train(self, dataloader):
        best_iid_val_loss = float('inf')
        counter = 0
        patience = 7
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                self.optimizer.zero_grad()
                output = self.model(smiles, proteins)
                output = output.squeeze(1)
                loss = F.mse_loss(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())

            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
            print_result(epoch, train_loss, val_loss, val_acc)
            self.scheduler.step(val_loss)
            if val_loss < best_iid_val_loss:
                best_iid_val_loss = val_loss
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
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                output = self.model(smiles, proteins)
                output = output.squeeze(1)
                loss = F.mse_loss(output, labels)
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())
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

    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = 7
        for epoch in range(self.num_epochs):
            self.model.train()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                self.optimizer.zero_grad()
                index = torch.randperm(smiles.size(0)).to(device)
                lambda_val = np.random.beta(0.4, 0.4)
                mixed_smiles = lambda_val * smiles + (1 - lambda_val) * smiles[index]
                mixed_proteins = lambda_val * proteins + (1 - lambda_val) * proteins[index]
                mixed_targets = lambda_val * labels + (1 - lambda_val) * labels[index]
                output = self.model(mixed_smiles.long(), mixed_proteins.long())
                output = output.squeeze(1)
                loss = F.mse_loss(output, mixed_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())
                running_loss += loss.item()

            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
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


class IRM(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, lambda_=1.0):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.lambda_ = lambda_

    def irm_penalty(self, model, outputs, labels, domain_ids):
        domain_losses = []
        for domain in torch.unique(domain_ids):
            domain_mask = (domain_ids == domain)
            domain_outputs = outputs[domain_mask]
            domain_labels = labels[domain_mask]
            domain_loss = F.mse_loss(domain_outputs, domain_labels)
            domain_losses.append(domain_loss)
        grad_params_1 = torch.autograd.grad(torch.stack(domain_losses[0::2]).mean(), model.parameters(),
                                            create_graph=True)
        grad_params_2 = torch.autograd.grad(torch.stack(domain_losses[1::2]).mean(), model.parameters(),
                                            create_graph=True)
        penalty = sum([(g1 * g2).sum() for g1, g2 in zip(grad_params_1, grad_params_2)])

        return penalty

    def train(self, dataloader):
        for epoch in range(self.num_epochs):
            self.model.train()
            all_predictions = []
            all_labels = []
            running_loss = 0.0
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                self.optimizer.zero_grad()
                output = self.model(smiles.long(), proteins.long()).squeeze(1)
                standard_loss = F.mse_loss(output, labels)

                penalty = self.irm_penalty(self.model, output, labels, domain_ids)

                total_loss = standard_loss + self.lambda_ * penalty
                total_loss.backward()
                self.optimizer.step()

                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())
                running_loss += total_loss.item()
            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
            print_result(epoch, train_loss, val_loss, val_acc)


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


    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = 3
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                self.optimizer.zero_grad()
                output = self.model(smiles, proteins)
                output = output.squeeze(1)
                if epoch / self.num_epochs < self.percentage:
                    loss = F.mse_loss(output, labels, reduction='none')
                else:
                    loss = F.mse_loss(output, labels)
                if epoch / self.num_epochs < self.percentage:
                    _, sorted_loss_index = torch.sort(loss, descending=True)
                    keep_size = int((1 - self.sample_ratio) * loss.size(0))
                    selected_indices = sorted_loss_index[:keep_size]
                    smiles = smiles[selected_indices]
                    proteins = proteins[selected_indices]
                    labels = labels[selected_indices]
                    cls_labels = cls_labels[selected_indices]

                    output = self.model(smiles, proteins)
                    output = output.squeeze(1)
                    loss = F.mse_loss(output, labels)

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                # Get the embeddings with get_features=True
                smiles_embeddings, proteins_embeddings = self.model(smiles, proteins, get_features=True)

                # Calculate gradients
                grad_smiles = torch.autograd.grad(outputs=smiles_embeddings, inputs=self.model.smiles_embedding.weight,
                                                  grad_outputs=torch.ones_like(smiles_embeddings), retain_graph=True)[0][smiles, :]
                grad_proteins = torch.autograd.grad(outputs=proteins_embeddings, inputs=self.model.proteins_embedding.weight,
                                    grad_outputs=torch.ones_like(proteins_embeddings), retain_graph=True)[0][proteins, :]

                # Identify top-k gradients to be masked out
                topk_smiles = [torch.topk(grad_smiles[i].norm(dim=-1), int(self.feature_ratio * grad_smiles.shape[1]),
                                          largest=True)[1] for i in range(grad_smiles.shape[0])]
                topk_proteins = [
                    torch.topk(grad_proteins[i].norm(dim=-1), int(self.feature_ratio * grad_proteins.shape[1]),
                               largest=True)[1] for i in range(grad_proteins.shape[0])]

                # Initialize masks
                mask_smiles = torch.ones_like(smiles_embeddings).to(device)
                mask_proteins = torch.ones_like(proteins_embeddings).to(device)

                idx_smiles = torch.stack(topk_smiles)
                idx_proteins = torch.stack(topk_proteins)
                # Set the positions corresponding to the top-k gradients to zero in the masks
                for i in range(mask_smiles.shape[0]):
                    mask_smiles[i, idx_smiles[i], :] = 0
                    mask_proteins[i, idx_proteins[i], :] = 0

                # Apply the masks to the embeddings
                masked_smiles_embeddings = smiles_embeddings * mask_smiles
                masked_proteins_embeddings = proteins_embeddings * mask_proteins

                # Call the forward_from_embeddings method with masked embeddings
                output = self.model.forward_from_embeddings(masked_smiles_embeddings, masked_proteins_embeddings)
                output = output.squeeze(1)
                loss = F.mse_loss(output, labels)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())

            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
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

        return val_acc

class DeepCoral(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, weight):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.weight = weight
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.08036246056856665,
                                                        patience=3, verbose=True)

    def coral_loss(self, source, target):
        source_coral = (source - torch.mean(source, 0)).t() @ (source - torch.mean(source, 0))
        target_coral = (target - torch.mean(target, 0)).t() @ (target - torch.mean(target, 0))
        loss = torch.norm(source_coral - target_coral, p='fro')
        return loss

    def train(self, dataloader):
        best_val_loss = float('inf')
        counter = 0
        patience = 10

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_coral_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []

            for (src_batch, tgt_batch) in zip(self.train_dataloader, self.ood_val_dataloader):
                src_smiles, src_proteins, src_labels, src_cls_labels, src_domain_ids = transfer_to_device(src_batch, device)
                tgt_smiles, tgt_proteins, tgt_labels, tgt_cls_labels, tgt_domain_ids = transfer_to_device(tgt_batch, device)
                self.optimizer.zero_grad()

                src_output = self.model(src_smiles.long(), src_proteins.long())
                tgt_output = self.model(tgt_smiles.long(), tgt_proteins.long())
                coral_loss = self.coral_loss(src_output, tgt_output)
                output = self.model(src_smiles.long(), src_proteins.long())
                output = output.squeeze(1)
                mse_loss = F.mse_loss(output, src_labels)
                coral_loss = coral_loss * self.weight
                total_loss = mse_loss

                total_loss.backward()
                self.optimizer.step()

                running_loss += mse_loss.item()
                running_coral_loss += coral_loss.item()

                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(src_cls_labels.cpu().numpy())

            acc, auc = self.compute_metrics(all_predictions, all_labels)
            train_loss = running_loss / len(dataloader)
            train_coral_loss = running_coral_loss / len(dataloader)
            print(f'Train Accuracy: {acc:.4f}, AUC: {auc:.4f}, MSE Loss: {train_loss:.4f}, CORAL Loss: {train_coral_loss:.12f}')

            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
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

class PGD(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, epsilons, epsilonp, alphas, alphap, num_iter, early_patience, learn_patience, factor):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.epsilons = epsilons
        self.epsilonp = epsilonp
        self.alphas = alphas
        self.alphap = alphap
        self.num_iter = num_iter
        self.early_patience = early_patience
        self.learn_patience = learn_patience
        self.factor = factor
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.factor, patience=self.learn_patience, verbose=True)

    def pgd_attack_on_embeddings(self, model, smiles_embeddings, protein_embeddings, labels, loss_func, epsilons, epsilonp, alphas, alphap, num_iter):
        emb_s_adv = smiles_embeddings.clone().detach().requires_grad_(True)
        emb_p_adv = protein_embeddings.clone().detach().requires_grad_(True)
        for i in range(num_iter):
            model.zero_grad()
            output = model.forward_from_embeddings(emb_s_adv, emb_p_adv).squeeze(1)
            loss = loss_func(output, labels)
            loss.backward()
            with torch.no_grad():
                adv_emb_s = emb_s_adv + alphas * emb_s_adv.grad.sign()
                eta_s = torch.clamp(adv_emb_s - smiles_embeddings, min=-epsilons, max=epsilons)
                emb_s_adv = torch.clamp(smiles_embeddings + eta_s, min=-1.7, max=2)
                emb_s_adv.requires_grad_(True)

                # compute adv update for protein_embeddings
                adv_emb_p = emb_p_adv + alphap * emb_p_adv.grad.sign()
                eta_p = torch.clamp(adv_emb_p - protein_embeddings, min=-epsilonp, max=epsilonp)
                emb_p_adv = torch.clamp(protein_embeddings + eta_p, min=-2.7, max=2)
                emb_p_adv.requires_grad_(True)

        return emb_s_adv.detach(), emb_p_adv.detach()

    def train(self, dataloader):
        best_model_state = None
        best_val_loss = float('inf')
        counter = 0
        patience = self.early_patience
        criterion = nn.MSELoss()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.model.train()
            all_predictions = []
            all_labels = []
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)
                self.optimizer.zero_grad()
                mixed_smiles = smiles
                mixed_proteins = proteins
                mixed_labels = labels
                mixed_smiles_emb, mixed_proteins_emb = self.model(mixed_smiles.long(), mixed_proteins.long(),
                                                                  get_features=True)
                outputs = self.model.forward_from_embeddings(mixed_smiles_emb, mixed_proteins_emb).squeeze(1)
                mixup_loss = criterion(outputs, mixed_labels)
                adv_mixed_smiles_emb, adv_mixed_proteins_emb = self.pgd_attack_on_embeddings(self.model,
                                                                                             mixed_smiles_emb,
                                                                                             mixed_proteins_emb,
                                                                                             mixed_labels, criterion, self.epsilons, self.epsilonp, self.alphas, self.alphap, self.num_iter)

                adv_outputs = self.model.forward_from_embeddings(adv_mixed_smiles_emb, adv_mixed_proteins_emb).squeeze(1)
                trades_loss = F.kl_div(F.log_softmax(outputs, dim=0), F.softmax(adv_outputs, dim=0), reduction='batchmean')
                combined_loss = 0.5 * mixup_loss + 0.5 * trades_loss
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                running_loss += combined_loss.item()
                all_predictions.extend(adv_outputs.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())

            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
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
        return val_acc

class BIOW2D(ERM):
    def __init__(self, model, dataset, num_epochs, optimizer, sample_ratio, feature_ratio, percentage, early_patience, learn_patience, factor, weight):
        super().__init__(model, dataset, num_epochs, optimizer)
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.percentage = percentage
        self.train_dataloader = dataset.dataloader_train
        self.early_patience = early_patience
        self.learn_patience = learn_patience
        self.factor = factor
        self.weight = weight
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.factor, patience=self.learn_patience, verbose=True)
        self.tokenizer_proteins = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
        self.tokenizer_smiles = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')

    def are_predictions_correct(self, predictions, labels):
        binary_predictions = (predictions >= self.threshold).astype(int)
        correct_predictions = binary_predictions == labels
        return correct_predictions

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
            epoch_correct_smiles_proteins_tokens = []
            for batch in dataloader:
                smiles, proteins, labels, cls_labels, domain_ids = transfer_to_device(batch, device)


                self.optimizer.zero_grad()
                output = self.model(smiles, proteins)
                output = output.squeeze(1)
                if epoch / self.num_epochs < self.percentage:
                    loss = F.mse_loss(output, labels, reduction='none')
                else:
                    loss = F.mse_loss(output, labels)
                if epoch / self.num_epochs < self.percentage:
                    _, sorted_loss_index = torch.sort(loss, descending=True)
                    keep_size = int((1 - self.sample_ratio) * loss.size(0))
                    selected_indices = sorted_loss_index[:keep_size]
                    smiles = smiles[selected_indices]
                    proteins = proteins[selected_indices]
                    labels = labels[selected_indices]
                    cls_labels = cls_labels[selected_indices]

                    output = self.model(smiles, proteins)
                    output = output.squeeze(1)
                    loss = F.mse_loss(output, labels)

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                grad_smiles = self.model.smiles_embedding.weight.grad[smiles, :].to(device)
                grad_proteins = self.model.proteins_embedding.weight.grad[proteins, :].to(device)


                topk_smiles = [torch.topk(grad_smiles[i].norm(dim=-1), int(self.feature_ratio * grad_smiles.shape[1]), largest=True)[1] for i in range(grad_smiles.shape[0])]
                topk_proteins = [torch.topk(grad_proteins[i].norm(dim=-1), int(self.feature_ratio * grad_proteins.shape[1]), largest=True)[1] for i in range(grad_proteins.shape[0])]

                smiles_mask_token_id = 0
                proteins_mask_token_id = 0
                masked_smiles = smiles.clone()
                masked_proteins = proteins.clone()
                if topk_smiles and topk_proteins:
                    for i in range(len(smiles)):
                        masked_smiles[i][topk_smiles[i]] = smiles_mask_token_id
                    for i in range(len(proteins)):
                        masked_proteins[i][topk_proteins[i]] = proteins_mask_token_id


                adv_output = self.model(masked_smiles, masked_proteins)
                adv_output = adv_output.squeeze(1)
                loss = F.mse_loss(output, labels)
                adversarial_loss = F.kl_div(F.log_softmax(output, dim=0), F.softmax(adv_output, dim=0), reduction='batchmean')
                combined_loss = loss + self.weight * adversarial_loss
                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(output.detach().cpu().numpy())
                all_labels.extend(cls_labels.cpu().numpy())
                # predictions_are_correct = self.are_predictions_correct(output.detach().cpu().numpy(), cls_labels.detach().cpu().numpy())
                # unique_proteins = set()  # Keep track of unique proteins
                # # print("Original:", self.tokenizer_smiles.decode(smiles[0].tolist()))
                # # print("Masked:", self.tokenizer_smiles.decode(masked_smiles[0].tolist()))
                # # high_gradient_tokens = smiles[0][topk_smiles[0]].tolist()
                # # print("Tokens with highest gradients:", self.tokenizer_smiles.decode(high_gradient_tokens))
                # for i, is_correct in enumerate(predictions_are_correct):
                #     if is_correct:
                #         # current_protein = self.tokenizer_proteins.decode(
                #         #     proteins[i].tolist())  # Decode the protein sequence
                #         if proteins[i] not in unique_proteins:  # Check if the protein is unique
                #             unique_proteins.add(proteins[i])
                #             epoch_correct_smiles_proteins_tokens.append({
                #                 # 'smiles': self.tokenizer_smiles.decode(smiles[i].tolist()),
                #                 'smiles': smiles[i],
                #                 # Decoding the smiles for better readability
                #                 'proteins': proteins[i],
                #                 'topk_smiles': smiles[i][topk_smiles[i]],
                #                 'topk_proteins': proteins[i][topk_proteins[i]]
                #             })
            train_loss = running_loss / len(dataloader)
            val_loss, val_acc = self.val_evaluate(self.ood_val_dataloader)
            print_result(epoch, train_loss, val_loss, val_acc)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_correct_smiles_proteins_tokens = epoch_correct_smiles_proteins_tokens
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), 'best_model.pth')
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered.")
                self.model.load_state_dict(best_model_state)
                break

        return val_acc, best_correct_smiles_proteins_tokens
