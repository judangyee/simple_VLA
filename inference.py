import torch
import torchvision.transforms as T
from PIL import Image

from Pytorch_Dataset import RobotLanguageDataset
from Training import MultimodalPolicy, Vocab

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pth"

# -----------------------
# Load checkpoint
# -----------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# Restore vocab
vocab = Vocab()
vocab.word2idx = ckpt["vocab"]
vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}
vocab_size = len(vocab.word2idx)

# Restore action map
action_map = ckpt["action_map"]
id_to_action = {v: k for k, v in action_map.items()}

# -----------------------
# Image transform
# -----------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# -----------------------
# Load model
# -----------------------
model = MultimodalPolicy(
    vocab_size=vocab_size,
    num_actions=len(action_map)
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# -----------------------
# Text encode
# -----------------------
def encode_text(text):
    tokens = text.lower().split()
    return torch.tensor(
        [vocab.word2idx.get(w, 0) for w in tokens],
        dtype=torch.long
    )

# -----------------------
# Inference function
# -----------------------
def infer(image_path, text):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    text_ids = encode_text(text).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image, text_ids)
        pred_id = logits.argmax(dim=1).item()

    return id_to_action[pred_id]

# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    img = "D:/Project/VLA_Project/dataset/images/000001.png"
    text = "pick the green cube"

    print("Text:", text)
    print("Predicted action:", infer(img, text))
