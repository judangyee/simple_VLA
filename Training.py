import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from Pytorch_Dataset import RobotLanguageDataset

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DEVICE = 'cpu'

# -----------------------------
# Text vocabulary (simple)
# -----------------------------
class Vocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0}
        self.idx2word = {0: '<pad>'}

    def build(self, texts):
        idx = 1
        for text in texts:
            for w in text.split():
                if w not in self.word2idx:
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
                    idx += 1

    def encode(self, text):
        return [self.word2idx[w] for w in text.split()]


# -----------------------------
# Model
# -----------------------------
class MultimodalPolicy(nn.Module):
    def __init__(self, vocab_size, num_actions, text_dim=64):
        super().__init__()

        # Image encoder (ResNet18)
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()  # 512-dim

        # Text encoder
        self.embedding = nn.Embedding(vocab_size, text_dim)

        # Policy head
        self.fc = nn.Sequential(
            nn.Linear(512 + text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, image, text_ids):
        img_feat = self.cnn(image)
        text_feat = self.embedding(text_ids).mean(dim=1)
        feat = torch.cat([img_feat, text_feat], dim=1)
        return self.fc(feat)


# -----------------------------
# Prepare dataset
# -----------------------------
dataset = RobotLanguageDataset(
    root_dir='dataset',
    labels_file='dataset/labels.txt'
)

texts = [t for _, t, _ in dataset.samples]
vocab = Vocab()
vocab.build(texts)

num_actions = len(dataset.action_map)

# Custom collate function

def collate_fn(batch):
    images, texts, actions = zip(*batch)

    images = torch.stack(images)
    actions = torch.stack(actions)

    encoded = [vocab.encode(t) for t in texts]
    max_len = max(len(e) for e in encoded)

    padded = []
    for e in encoded:
        padded.append(e + [0] * (max_len - len(e)))

    text_ids = torch.tensor(padded, dtype=torch.long)

    return images, text_ids, actions


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# -----------------------------
# Train setup
# -----------------------------
model = MultimodalPolicy(
    vocab_size=len(vocab.word2idx),
    num_actions=num_actions
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for images, text_ids, actions in loader:
        images = images.to(DEVICE)
        text_ids = text_ids.to(DEVICE)
        actions = actions.to(DEVICE)

        logits = model(images, text_ids)
        loss = criterion(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == actions).sum().item()
        total += actions.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Acc: {acc:.2f}%")

print("Training finished")

# -----------------------------
# Save trained model
# -----------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab.word2idx,
    "action_map": dataset.action_map
}, "model.pth")

print("Model saved to model.pth")
