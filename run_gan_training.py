import json
import os
import random
from pathlib import Path

import cv2
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DATA_ROOT = Path('d:/BaiduNetdiskDownload/dataset_work_final')
TRAIN_IMG_DIR = DATA_ROOT / 'images' / 'train'
TEST_IMG_DIR = DATA_ROOT / 'images' / 'test'
LABEL_DIR = DATA_ROOT / 'labels' / 'train'
CHECKPOINT_DIR = DATA_ROOT / 'artifacts_gan'
RESULT_DIR = DATA_ROOT / 'results_cgan'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Oil_accumulation', 'Oil_seepage', 'Standing_water']
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (288, 512)
SEED = 42
FAST_TRAIN_LIMIT = 64
FAST_VAL_LIMIT = 16
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-4
LAMBDA_L1 = 50.0
LAMBDA_DICE = 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
PIN_MEMORY = DEVICE == 'cuda'

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

seed_everything()
print('device:', DEVICE)


def load_yolo_boxes(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            cls = int(cls)
            if cls >= NUM_CLASSES:
                continue
            boxes.append((cls, cx, cy, w, h))
    return boxes


def boxes_to_mask(boxes, out_h, out_w):
    mask = np.zeros((out_h, out_w, NUM_CLASSES), dtype=np.float32)
    for cls, cx, cy, bw, bh in boxes:
        x1 = max(0, int((cx - bw / 2) * out_w))
        x2 = min(out_w, int((cx + bw / 2) * out_w))
        y1 = max(0, int((cy - bh / 2) * out_h))
        y2 = min(out_h, int((cy + bh / 2) * out_h))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2, cls] = 1.0
    return mask


def apply_augmentations(image, mask):
    if random.random() < 0.5:
        image = image[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    if random.random() < 0.3:
        alpha = 1.0 + 0.4 * (random.random() * 2 - 1)
        beta = 20.0 * (random.random() * 2 - 1)
        image = np.clip(image * alpha + beta, 0, 255)
    if random.random() < 0.2:
        noise = np.random.normal(0, 10, size=image.shape)
        image = np.clip(image + noise, 0, 255)
    return image, mask


class LeakDataset(Dataset):
    def __init__(self, image_ids, augment=False):
        self.image_ids = image_ids
        self.augment = augment

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = TRAIN_IMG_DIR / f'{image_id}.jpg'
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        boxes = load_yolo_boxes(LABEL_DIR / f'{image_id}.txt')
        mask = boxes_to_mask(boxes, IMAGE_SIZE[0], IMAGE_SIZE[1])
        if self.augment:
            image, mask = apply_augmentations(image, mask)
        image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
        mask = torch.from_numpy(mask.transpose(2,0,1)).float()
        return image, mask


class LeakInferenceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        tensor = torch.from_numpy(image_resized.transpose(2,0,1)).float() / 255.0
        return tensor, path.stem, (orig_h, orig_w)


all_ids = sorted([p.stem for p in TRAIN_IMG_DIR.glob('*.jpg')])
random.Random(SEED).shuffle(all_ids)
val_ids = sorted(all_ids[:FAST_VAL_LIMIT])
train_ids = sorted(all_ids[FAST_VAL_LIMIT:FAST_VAL_LIMIT+FAST_TRAIN_LIMIT])
print('train samples:', len(train_ids), 'val samples:', len(val_ids))

with (CHECKPOINT_DIR / 'split_gan.json').open('w', encoding='utf-8') as f:
    json.dump({'train_ids': train_ids, 'val_ids': val_ids}, f, ensure_ascii=False, indent=2)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=NUM_CLASSES, base_ch=64):
        super().__init__()
        self.down1 = ConvBlock(in_ch, base_ch, use_norm=False)
        self.down2 = ConvBlock(base_ch, base_ch*2)
        self.down3 = ConvBlock(base_ch*2, base_ch*4)
        self.down4 = ConvBlock(base_ch*4, base_ch*8)
        self.down5 = ConvBlock(base_ch*8, base_ch*8)
        self.down6 = ConvBlock(base_ch*8, base_ch*8)
        self.down7 = ConvBlock(base_ch*8, base_ch*8)
        self.bottom = ConvBlock(base_ch*8, base_ch*8)
        self.up1 = DeconvBlock(base_ch*8, base_ch*8, dropout=True)
        self.up2 = DeconvBlock(base_ch*16, base_ch*8, dropout=True)
        self.up3 = DeconvBlock(base_ch*16, base_ch*8, dropout=True)
        self.up4 = DeconvBlock(base_ch*16, base_ch*8)
        self.up5 = DeconvBlock(base_ch*16, base_ch*4)
        self.up6 = DeconvBlock(base_ch*8, base_ch*2)
        self.up7 = DeconvBlock(base_ch*4, base_ch)
        self.final = nn.ConvTranspose2d(base_ch*2, out_ch, 4, stride=2, padding=1)

    @staticmethod
    def match_size(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bott = self.bottom(d7)
        u1 = self.up1(bott)
        u1 = torch.cat([self.match_size(u1, d7), d7], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([self.match_size(u2, d6), d6], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([self.match_size(u3, d5), d5], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([self.match_size(u4, d4), d4], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([self.match_size(u5, d3), d3], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([self.match_size(u6, d2), d2], dim=1)
        u7 = self.up7(u6)
        u7 = torch.cat([self.match_size(u7, d1), d1], dim=1)
        out = self.final(u7)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3 + NUM_CLASSES, base_ch=64):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_ch, base_ch, use_norm=False),
            ConvBlock(base_ch, base_ch*2),
            ConvBlock(base_ch*2, base_ch*4),
            nn.Conv2d(base_ch*4, 1, 4, stride=1, padding=1)
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)


generator = UNetGenerator().to(DEVICE)
discriminator = PatchDiscriminator().to(DEVICE)
print('Generator params:', sum(p.numel() for p in generator.parameters())/1e6, 'M')

adv_criterion = nn.BCEWithLogitsLoss()


def dice_loss(preds, targets, eps=1e-6):
    probs = torch.sigmoid(preds)
    dims = (0,2,3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2*intersection + eps)/(union + eps)
    return 1 - dice.mean()


def generator_loss(gen_out, real_imgs, real_masks):
    fake_disc = discriminator(real_imgs, gen_out)
    valid = torch.ones_like(fake_disc)
    adv = adv_criterion(fake_disc, valid)
    l1 = F.l1_loss(gen_out, real_masks)
    dloss = dice_loss(gen_out, real_masks)
    return adv + LAMBDA_L1 * l1 + LAMBDA_DICE * dloss


def discriminator_loss(real_imgs, real_masks, fake_masks):
    valid = torch.ones_like(discriminator(real_imgs, real_masks))
    fake = torch.zeros_like(valid)
    real_pred = discriminator(real_imgs, real_masks)
    fake_pred = discriminator(real_imgs, fake_masks.detach())
    return 0.5 * (adv_criterion(real_pred, valid) + adv_criterion(fake_pred, fake))

train_dataset = LeakDataset(train_ids, augment=True)
val_dataset = LeakDataset(val_ids, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
print('train batches:', len(train_loader), 'val batches:', len(val_loader))

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))

history = []
for epoch in range(1, EPOCHS + 1):
    generator.train(); discriminator.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
    total_g = total_d = 0.0
    for images, masks in pbar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        fake_masks = generator(images)
        optimizer_d.zero_grad()
        d_loss = discriminator_loss(images, masks, fake_masks)
        d_loss.backward()
        optimizer_d.step()
        optimizer_g.zero_grad()
        g_loss = generator_loss(fake_masks, images, masks)
        g_loss.backward()
        optimizer_g.step()
        total_g += g_loss.item()
        total_d += d_loss.item()
        pbar.set_postfix({'g': g_loss.item(), 'd': d_loss.item()})
    generator.eval()
    val_l1 = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = generator(images)
            val_l1 += F.l1_loss(preds, masks).item()
            val_dice += (1 - dice_loss(preds, masks)).item()
    val_l1 /= max(1,len(val_loader))
    val_dice /= max(1,len(val_loader))
    rec = {'epoch': epoch, 'g_loss': total_g/len(train_loader), 'd_loss': total_d/len(train_loader), 'val_l1': val_l1, 'val_dice': val_dice}
    history.append(rec)
    print(rec)
    torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(), 'epoch': epoch}, CHECKPOINT_DIR / 'cgan_detector.pt')

print('history:', history)

# inference
for old in RESULT_DIR.glob('*.txt'):
    old.unlink()

test_dataset = LeakInferenceDataset(TEST_IMG_DIR.glob('*.jpg'))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def _to_list(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def normalize_original_sizes(batch_sizes):
    if isinstance(batch_sizes, (list, tuple)):
        norm = []
        for size in batch_sizes:
            vals = _to_list(size)
            if len(vals) == 1 and isinstance(vals[0], (list, tuple)):
                vals = vals[0]
            if len(vals) != 2:
                vals = [vals[0], vals[0]] if len(vals) == 1 else vals[:2]
            norm.append((int(vals[0]), int(vals[1])))
        return norm
    vals = _to_list(batch_sizes)
    if len(vals) == 2:
        return [(int(vals[0]), int(vals[1]))]
    return [(int(vals[0]), int(vals[0]))]

with torch.no_grad():
    for images, image_ids, original_sizes in tqdm(test_loader, desc='test'):
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        else:
            image_ids = list(image_ids)
        original_sizes = normalize_original_sizes(original_sizes)
        images = images.to(DEVICE)
        preds = generator(images)
        probs = torch.sigmoid(preds).cpu().numpy()
        for prob, image_id, (orig_h, orig_w) in zip(probs, image_ids, original_sizes):
            resized = np.stack([cv2.resize(prob[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) for c in range(prob.shape[0])])
            boxes = []
            channels, h, w = resized.shape
            for cls_idx in range(channels):
                cls_map = resized[cls_idx]
                heat = cv2.GaussianBlur(cls_map, (5,5), 0)
                mask = (heat >= 0.45).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw * bh < 1e-4 * h * w:
                        continue
                    score = float(heat[y:y+bh, x:x+bw].mean())
                    boxes.append([cls_idx, score, x, y, bw, bh])
            boxes.sort(key=lambda x: x[1], reverse=True)
            keep = []
            while boxes:
                cur = boxes.pop(0)
                keep.append(cur)
                def _iou(a, b):
                    xa1, ya1 = a[2], a[3]
                    xa2, ya2 = xa1 + a[4], ya1 + a[5]
                    xb1, yb1 = b[2], b[3]
                    xb2, yb2 = xb1 + b[4], yb1 + b[5]
                    inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
                    return inter / (a[4]*a[5] + b[4]*b[5] - inter + 1e-6)
                boxes = [b for b in boxes if _iou(cur, b) < 0.4]
            with (RESULT_DIR / f'{image_id}.txt').open('w') as f:
                for cls, score, x, y, bw, bh in keep:
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}\n")
print('result files:', len(list(RESULT_DIR.glob('*.txt'))))

