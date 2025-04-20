from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,2), (2,1), (0,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,2), (2,1), (0,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()
        )

        self.rnn1 = nn.LSTM(512, nh, bidirectional=True)
        self.rnn2 = nn.LSTM(nh * 2, nh, bidirectional=True)

        self.linear = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # [B, C, W]
        conv = conv.permute(2, 0, 1)  # [W, B, C]

        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)

        output = self.linear(recurrent)  # [W, B, num_classes]
        return output



import string
CHARS = string.digits + string.ascii_uppercase  # '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_idx = {char: idx+1 for idx, char in enumerate(CHARS)}  # +1 for CTC blank
idx_to_char = {idx+1: char for idx, char in enumerate(CHARS)}
nclass = len(CHARS) + 1  # +1 for blank



class PlateDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(label_file, 'r') as f:
            lines = f.readlines()
        self.samples = [(line.split()[0], line.strip().split()[1]) for line in lines]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      img_name, label = self.samples[idx]
      img_path = os.path.join(self.image_folder, img_name)
      image = Image.open(img_path).convert('L')
      if self.transform:
        image = self.transform(image)
      label_idx = [char_to_idx[c] for c in label if c in char_to_idx]
      return image, torch.tensor(label_idx), len(label_idx) 




transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = PlateDataset("/content/Final_dataset/cropped_plates", "/content/Final_dataset/labels.txt", transform)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)


model = CRNN(imgH=32, nc=1, nclass=nclass, nh=256).to("cuda")



device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def collate_batch(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)

def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels, lengths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)  # [W, B, num_classes]
            output_log_probs = F.log_softmax(output, dim=2)

            input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long).to(device)
            loss = criterion(output_log_probs, labels, input_lengths, lengths)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

train(model, train_loader, criterion, optimizer, epochs=100)


