import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RobotLanguageDataset(Dataset):
    """
    Dataset for (image, language, action) tuples
    labels.txt format:
    image_name | command text | action label
    """

    def __init__(self, root_dir, labels_file, action_map=None, transform=None):
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.transform = transform

        self.samples = []

        with open(labels_file, 'r') as f:
            for line in f:
                img, text, action = line.strip().split('|')
                self.samples.append((img.strip(), text.strip(), action.strip()))

        # action string -> integer id
        if action_map is None:
            actions = sorted(list(set(a for _, _, a in self.samples)))
            self.action_map = {a: i for i, a in enumerate(actions)}
        else:
            self.action_map = action_map

        # basic image transform
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text, action = self.samples[idx]

        img_path = os.path.join(self.root_dir, 'images', img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        action_id = self.action_map[action]
        action_id = torch.tensor(action_id, dtype=torch.long)

        return image, text, action_id

# -----------------------------
# Simple test
# -----------------------------
if __name__ == '__main__':
    dataset = RobotLanguageDataset(
        root_dir='dataset',
        labels_file='dataset/labels.txt'
    )

    img, text, action = dataset[0]
    print('Image:', img.shape)
    print('Text:', text)
    print('Action:', action)