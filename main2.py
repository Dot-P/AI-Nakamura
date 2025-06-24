import os
import shutil
import subprocess
import torch
import clip
from PIL import Image
from mylib.data_io import CSVBasedDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 1) データセット準備
DATA_DIR = "./Dataset"
EXTRACT_DIR = "FakeFaces"
URL = "https://tus.box.com/shared/static/j2j32h27j21c2xi610xer0ar8hcsq34t.gz"
TAR_NAME = "FakeFaces.tar.gz"

def prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isfile(f"{DATA_DIR}/FakeFaces_train_images.pt"):
        return
    subprocess.run(["wget", URL, "-O", TAR_NAME], check=True)
    subprocess.run(["tar", "-zxf", TAR_NAME], check=True)
    os.remove(TAR_NAME)

    def make_tensors(csv_name):
        ds = CSVBasedDataset(
            dirname=EXTRACT_DIR,
            filename=os.path.join(EXTRACT_DIR, csv_name),
            items=["File Path", "Label"],
            dtypes=["image", "label"]
        )
        data = [ds[i] for i in range(len(ds))]
        images = torch.cat([u.unsqueeze(0) for u, _ in data], dim=0)
        labels = torch.cat([v.unsqueeze(0) for _, v in data], dim=0)
        return images, labels

    for split in ["train", "test"]:
        imgs, lbls = make_tensors(f"{split}_list.csv")
        torch.save(imgs, f"{DATA_DIR}/FakeFaces_{split}_images.pt")
        torch.save(lbls, f"{DATA_DIR}/FakeFaces_{split}_labels.pt")
    shutil.rmtree(EXTRACT_DIR)

# 2) Dataset: PT読み込み＋CLIP前処理
class FakeFacesDataset(Dataset):
    def __init__(self, imgs_pt, labels_pt, preprocess):
        self.images = torch.load(imgs_pt)
        self.labels = torch.load(labels_pt)
        self.pre = preprocess
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        return self.pre(img), int(self.labels[idx])

# 3) モデル: CLIP凍結＋テキスト特徴固定＋学習可能MLPヘッド
class DeepFakeClipFinetune(nn.Module):
    def __init__(self, clip_model, prompts, device, hidden_dim=256):
        super().__init__()
        self.device = device
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            txt_feats = self.clip.encode_text(tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        self.text_features = txt_feats
        sample = next(self.clip.parameters())
        self.proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 512)
        ).to(device=sample.device, dtype=sample.dtype)
        self.logit_scale = self.clip.logit_scale.exp()

    def forward(self, x):
        with torch.no_grad():
            feats = self.clip.encode_image(x)
        feats = feats.to(self.proj[0].weight.dtype)
        feats = self.proj(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        dtype = feats.dtype
        txt = self.text_features.to(dtype)
        scale = self.logit_scale.to(dtype)
        return scale * feats @ txt.T

# 4) 学習・評価関数

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    # 進捗バーはこのループのみ
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(model.device), labels.to(model.device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_loss_acc(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    # 進捗バーを除外
    for imgs, labels in loader:
        imgs, labels = imgs.to(model.device), labels.to(model.device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

# 5) メイン
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepare_dataset()
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    prompts = ["a human face", "a deepfake face"]

    train_ds = FakeFacesDataset(
        f"{DATA_DIR}/FakeFaces_train_images.pt",
        f"{DATA_DIR}/FakeFaces_train_labels.pt",
        preprocess
    )
    val_ds = FakeFacesDataset(
        f"{DATA_DIR}/FakeFaces_test_images.pt",
        f"{DATA_DIR}/FakeFaces_test_labels.pt",
        preprocess
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = DeepFakeClipFinetune(clip_model, prompts, device).to(device)
    optimizer = optim.Adam(model.proj.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 40
    for ep in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        train_acc = eval_loss_acc(model, train_loader, criterion)[1]
        val_loss, val_acc = eval_loss_acc(model, val_loader, criterion)
        print(f"[Epoch {ep}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

