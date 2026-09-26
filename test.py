import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path  = os.path.abspath(os.path.join(script_dir, ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.models import Baseline


def load_model(model_path, embedding_dim=512, device="cpu"):
    model = Baseline(embedding_dim=embedding_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model carregat des de: {model_path}")
    return model


def test(model, test_loader, device="cuda", log_wandb=False):
    model.eval()
    all_txt, all_img = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images         = batch['image'].to(device)

            txt_emb, img_emb = model(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                images
            )
            all_txt.append(txt_emb)
            all_img.append(img_emb)

    txts = torch.cat(all_txt)   
    imgs = torch.cat(all_img)   

    sim = torch.matmul(txts, imgs.t())

    indices_ordenats = torch.argsort(sim, dim=1, descending=True)
    ground_truth = torch.arange(len(txts), device=device).view(-1, 1)

    rangs = (indices_ordenats == ground_truth).nonzero()[:, 1] + 1

    r1  = (sim.topk(1,  dim=1)[1] == ground_truth).any(dim=1).float().mean().item()
    r10 = (sim.topk(10, dim=1)[1] == ground_truth).any(dim=1).float().mean().item()
    mrr = (1.0 / rangs.float()).mean().item()

    print(f"\n{'='*40}")
    print(f"  Test R@1  : {r1:.4f}")
    print(f"  Test R@10 : {r10:.4f}")
    print(f"  Test MRR  : {mrr:.4f}")
    print(f"{'='*40}\n")

    if log_wandb:
        wandb.log({"test/R@1": r1, "test/R@10": r10, "test/MRR": mrr})

    return {"R@1": r1, "R@10": r10, "MRR": mrr}


def demo(model, image_path, texts, device="cpu"):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    encoding = model.sbert_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    text_features = {k: v.to(device) for k, v in encoding.items()}

    model.eval()
    with torch.no_grad():
        txt_emb, img_emb = model(
            text_features,
            image_tensor.expand(len(texts), -1, -1, -1)
        )

    similarities = F.cosine_similarity(txt_emb, img_emb, dim=1)

    print(f"\nImatge: {image_path}")
    print(f"{'─'*50}")
    for text, sim in zip(texts, similarities):
        print(f"  [{sim:.4f}]  {text}")

    best_idx = similarities.argmax().item()
    print(f"{'─'*50}")
    print(f"  ✓ Millor coincidència: \"{texts[best_idx]}\" (sim={similarities[best_idx]:.4f})\n")

    return {text: sim.item() for text, sim in zip(texts, similarities)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test i demo del model Comics Retrieval")

    parser.add_argument("--model_path",    type=str, required=True,
                        help="Ruta al fitxer .pth del model entrenat")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Dimensió de l'embedding (per defecte: 512)")
    parser.add_argument("--device",        type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Dispositiu de càlcul (per defecte: cpu)")

    parser.add_argument("--image", type=str, default=None,
                        help="[Demo] Ruta a la imatge d'exemple")
    parser.add_argument("--texts", type=str, nargs="+", default=None,
                        help="[Demo] Textos candidats separats per espai")

    parser.add_argument("--eval",  action="store_true",
                        help="[Eval] Avalua sobre el dataset de test complet")
    parser.add_argument("--wandb", action="store_true",
                        help="[Eval] Envia els resultats a Weights & Biases")

    args = parser.parse_args()

    model = load_model(args.model_path, args.embedding_dim, args.device)

    if args.image and args.texts:
        demo(model, args.image, args.texts, device=args.device)

    if args.eval:
        from datasets import load_dataset
        from dataloaders.dataset import ComicsPAPDataset

        raw_dataset = load_dataset("/home/datasets/COMIC-PAP")
        test_data   = ComicsPAPDataset(raw_dataset['validation'], model.sbert_tokenizer)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=32, shuffle=False
        )

        if args.wandb:
            wandb.init(project="Comics-Retrieval-Test")

        test(model, test_loader, device=args.device, log_wandb=args.wandb)

    if not args.image and not args.eval:
        print("Indica --image + --texts per al mode demo, o --eval per avaluar el dataset complet.")
        parser.print_help()