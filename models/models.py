import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms


from transformers import AutoModel, AutoTokenizer


class Baseline(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        freeze_sbert=True,
        freeze_resnet=True,
    ):

        super(Baseline, self).__init__()

        self.sbert = AutoModel.from_pretrained(sbert_model)
        self.sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_model)

        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False

        # ── Encoder d'imatge: ResNet50 ──
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Dimensions de sortida
        self.sbert_output_dim = self.sbert.config.hidden_size
        self.resnet_output_dim = 2048

        self.text_projection = nn.Sequential(
            nn.Linear(self.sbert_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(self.resnet_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    def _mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


    def preprocess_text(self, text_samples):
        encoded = self.sbert_tokenizer(
            text_samples,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return encoded

    def preprocess_image(self, image_paths):
        images = []
        for img in image_paths:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if isinstance(img, Image.Image):
                img = self.image_transform(img)
            images.append(img)
        return torch.stack(images)


    def encode_text(self, text_features=None, raw_text=None):

        if raw_text is not None:
            text_features = self.preprocess_text(raw_text).to(
                next(self.parameters()).device
            )

        sbert_frozen = not any(p.requires_grad for p in self.sbert.parameters())
        if sbert_frozen:
            with torch.no_grad():
                outputs = self.sbert(**text_features)
                sentence_emb = self._mean_pooling(outputs, text_features["attention_mask"]).detach()
        else:
            outputs = self.sbert(**text_features)
            sentence_emb = self._mean_pooling(outputs, text_features["attention_mask"])

        text_embeddings = self.text_projection(sentence_emb)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        return text_embeddings

    def encode_image(self, image_features=None, image_paths=None):
        if image_paths is not None:
            image_features = self.preprocess_image(image_paths)

        resnet_frozen = not any(p.requires_grad for p in self.resnet.parameters())
        if resnet_frozen:
            with torch.no_grad():
                feats = self.resnet(image_features)
                feats = feats.flatten(start_dim=1).detach()
        else:
            feats = self.resnet(image_features)
            feats = feats.flatten(start_dim=1)

        image_embeddings = self.image_projection(feats)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        return image_embeddings

    def forward(self, text_features=None, image_features=None,
                raw_text=None, image_paths=None):
        text_embeddings = None
        image_embeddings = None

        if text_features is not None or raw_text is not None:
            text_embeddings = self.encode_text(text_features, raw_text)

        if image_features is not None or image_paths is not None:
            image_embeddings = self.encode_image(image_features, image_paths)

        return text_embeddings, image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):

        return torch.mm(text_embeddings, image_embeddings.t())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Baseline(embedding_dim=512)
    model.to(device)
    model.eval()

    texts = [
        "A superhero fights a villain in the city",
        "Two characters are having a conversation",
        "An explosion destroys a building",
    ]
    images = torch.randn(3, 3, 224, 224).to(device)

    with torch.no_grad():
        text_emb, _ = model(raw_text=texts)
        _, img_emb  = model(image_features=images)

    sim = model.compute_similarity(text_emb, img_emb)
    print("Similarity matrix (Text x Images):")
    print(sim.cpu().numpy())
    print(f"\nText embedding dim: {text_emb.shape}")
    print(f"Image embedding dim: {img_emb.shape}")