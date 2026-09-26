from tqdm.auto import tqdm
import wandb
import torch
import os


def get_checkpoint_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "trained_models")
    return os.path.join(save_dir, "model_best.pth")


def unfreeze_last_layers(model):

    sbert_trainable = sum(1 for p in model.sbert.parameters() if p.requires_grad)
    if sbert_trainable == 0:
        print("SBERT completament congelat (recomanació professor)")
    else:
        print(f"SBERT parcialment entrenable ({sbert_trainable} paràmetres)")

    print("Descongelant layer4 de ResNet50 (índex 7 del Sequential)")
    for param in model.resnet[7].parameters():
        param.requires_grad = True


def save_checkpoint(state, is_best):
    if is_best:
        path = get_checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Millor model guardat a: {path} (R@1: {state['best_r1']:.4f})")


def train(model, loader, val_loader, criterion, optimizer, config, device="cuda") -> float:
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    batch_ct   = 0
    best_r1    = 0.0
    best_r10   = 0.0
    best_mrr   = 0.0

    unfreeze_last_layers(model)
    print("Capes finals descongelades. Iniciant entrenament.")

    for epoch in tqdm(range(config.epochs)):
        model.train()
        epoch_loss = 0.0

        for _, batch in enumerate(loader):
            images    = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)

            loss = train_batch(images, input_ids, mask, model, optimizer, criterion, device)

            example_ct += len(images)
            batch_ct   += 1
            epoch_loss += loss.item()

            if ((batch_ct + 1) % 25) == 0:
                wandb.log(
                    {"train/loss": loss.item(), "train/examples": example_ct},
                    step=batch_ct
                )
                print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

        val_metrics = validate_retrieval(model, val_loader, device)

        is_best_r1 = val_metrics["R@1"] > best_r1
        if is_best_r1:
            best_r1 = val_metrics["R@1"]
            path = get_checkpoint_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "best_r1":  best_r1,
            }, path)
            print(f"  Nou millor model guardat (R@1={best_r1:.4f})")

        if val_metrics["R@10"] > best_r10:
            best_r10 = val_metrics["R@10"]

        if val_metrics["MRR"] > best_mrr:
            best_mrr = val_metrics["MRR"]

        wandb.log({
            "epoch":             epoch,
            "train/epoch_loss":  epoch_loss / len(loader),
            "val/R@1":           val_metrics["R@1"],
            "val/R@10":          val_metrics["R@10"],
            "val/MRR":           val_metrics["MRR"],
            "val/best_R@1":      best_r1,
            "val/best_R@10":     best_r10,
            "val/best_MRR":      best_mrr,
        }, step=batch_ct)


    wandb.run.summary["val/best_R@1"]  = best_r1
    wandb.run.summary["val/best_R@10"] = best_r10
    wandb.run.summary["val/best_MRR"]  = best_mrr

    return best_r1


def train_batch(images, input_ids, mask, model, optimizer, criterion, device="cuda"):

    model.train()

    text_features = {
        'input_ids':      input_ids.squeeze(1),
        'attention_mask': mask.squeeze(1)
    }
    txt_emb, img_emb = model(text_features, images)

    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)

    N = txt_emb.size(0)

    indices    = torch.arange(N, device=device)
    embeddings = torch.cat([txt_emb, img_emb], dim=0)   
    labels     = torch.cat([indices, indices], dim=0)    

    loss = criterion(embeddings, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validate_retrieval(model, loader, device):

    model.eval()
    all_txt, all_img = [], []

    with torch.no_grad():
        for batch in loader:
            t, i = model(
                {
                    'input_ids':      batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                },
                batch['image'].to(device)
            )
            all_txt.append(t)
            all_img.append(i)

    txts = torch.nn.functional.normalize(torch.cat(all_txt), p=2, dim=1)
    imgs = torch.nn.functional.normalize(torch.cat(all_img), p=2, dim=1)

    sim  = torch.matmul(txts, imgs.t())
    gt   = torch.arange(len(txts), device=device).view(-1, 1)

    r1  = (sim.topk(1,  dim=1)[1] == gt).any(dim=1).float().mean().item()
    r10 = (sim.topk(min(10, len(txts)), dim=1)[1] == gt).any(dim=1).float().mean().item()

    ranks = (sim.argsort(dim=1, descending=True) == gt).nonzero(as_tuple=False)[:, 1]
    mrr   = (1.0 / (ranks.float() + 1)).mean().item()

    print(f"  val/R@1={r1:.4f}  val/R@10={r10:.4f}  val/MRR={mrr:.4f}")
    return {"R@1": r1, "R@10": r10, "MRR": mrr}