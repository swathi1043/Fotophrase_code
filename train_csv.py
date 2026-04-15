

import os
import re
import time
import pickle
import pandas as pd
from collections import Counter
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# ══════════════════════════════════════════════════════════════════
#   PATHS  
# ══════════════════════════════════════════════════════════════════

USERNAME    = "swath"   

BASE_DIR    = rf"C:\Users\{USERNAME}\Downloads\archive (2)\flickr30k_images"
IMAGES_DIR  = os.path.join(BASE_DIR, "flickr30k_images")   # folder with .jpg files
CSV_PATH    = os.path.join(BASE_DIR, "results.csv")         # captions file
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")         # where model is saved
VOCAB_PATH  = os.path.join(CKPT_DIR, "vocab.pkl")

# ══════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════

EMBED_SIZE     = 256
DECODER_DIM    = 512
ATTENTION_DIM  = 256
DROPOUT        = 0.5
FREQ_THRESHOLD = 5
EPOCHS         = 10
BATCH_SIZE     = 16
ENCODER_LR     = 1e-4
DECODER_LR     = 4e-4
GRAD_CLIP      = 5.0
ALPHA_C        = 1.0
PRINT_EVERY    = 50
NUM_WORKERS    = 0       # must be 0 on Windows


# ══════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        return re.sub(r"[^a-zA-Z0-9 ]", "", str(text).lower()).split()

    def build(self, captions):
        for cap in captions:
            self.word_freq.update(self.tokenize(cap))
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"  Vocabulary size: {len(self)} tokens")

    def encode(self, caption):
        tokens = self.tokenize(caption)
        return (
            [self.word2idx[self.SOS]]
            + [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
            + [self.word2idx[self.EOS]]
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Vocabulary saved → {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class Flickr30kDataset(Dataset):
    def __init__(self, images_dir, csv_path, vocab, split="train",
                 val_split=0.1, test_split=0.05, seed=42):
        self.images_dir = images_dir
        self.vocab = vocab

        # Transform
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

        # Load CSV
        df = self._load_csv(csv_path)

        # Split by unique images (no data leakage)
        import random
        unique_imgs = df["image_name"].unique().tolist()
        random.Random(seed).shuffle(unique_imgs)
        n = len(unique_imgs)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)

        if split == "test":
            selected = set(unique_imgs[:n_test])
        elif split == "val":
            selected = set(unique_imgs[n_test: n_test + n_val])
        else:
            selected = set(unique_imgs[n_test + n_val:])

        df_split = df[df["image_name"].isin(selected)]

        self.samples = []
        missing = 0
        for _, row in df_split.iterrows():
            img_path = os.path.join(images_dir, row["image_name"])
            if os.path.exists(img_path):
                self.samples.append((img_path, row["comment"]))
            else:
                missing += 1

        print(f"  [{split:5s}] {len(self.samples):>6} samples "
              f"| {len(selected)} images | {missing} missing files")

    @staticmethod
    def _load_csv(csv_path):
        # Try pipe separator (standard Kaggle Flickr30k format)
        try:
            df = pd.read_csv(csv_path, sep="|", engine="python")
            df.columns = [c.strip() for c in df.columns]
            if len(df.columns) == 3:
                df.columns = ["image_name", "comment_number", "comment"]
        except Exception:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]

        df = df.dropna(subset=["comment"])
        df["comment"]    = df["comment"].astype(str).str.strip()
        df["image_name"] = df["image_name"].astype(str).str.strip()
        print(f"  Loaded {len(df)} caption rows from CSV")
        return df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return image, encoded, len(encoded)


def collate_fn(batch):
    images, captions, lengths = zip(*batch)
    images = torch.stack(images, 0)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    lengths_t, sort_idx = lengths_t.sort(descending=True)
    images   = images[sort_idx]
    captions = [captions[i] for i in sort_idx]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded, lengths_t.tolist()


# ══════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

    def forward(self, images):
        with torch.no_grad():
            f = self.resnet(images)
        f = self.adaptive_pool(f)
        f = f.permute(0, 2, 3, 1)
        B, H, W, C = f.shape
        f = f.view(B, H * W, C)
        f = self.fc(f)
        return f


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att    = nn.Linear(attention_dim, 1)
        self.relu        = nn.ReLU()
        self.softmax     = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1  = self.encoder_att(encoder_out)
        att2  = self.decoder_att(decoder_hidden).unsqueeze(1)
        att   = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        ctx   = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return ctx, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, decoder_dim, vocab_size,
                 encoder_dim=256, attention_dim=256, dropout=0.5):
        super().__init__()
        self.vocab_size  = vocab_size
        self.decoder_dim = decoder_dim
        self.attention   = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding   = nn.Embedding(vocab_size, embed_size)
        self.dropout     = nn.Dropout(p=dropout)
        self.lstm_cell   = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.init_h      = nn.Linear(encoder_dim, decoder_dim)
        self.init_c      = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta      = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid     = nn.Sigmoid()
        self.fc          = nn.Linear(decoder_dim, vocab_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight,        -0.1, 0.1)
        nn.init.constant_(self.fc.bias,          0)

    def _init_hidden(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def forward(self, encoder_out, captions, lengths):
        B = encoder_out.size(0)
        embeddings = self.dropout(self.embedding(captions))
        h, c = self._init_hidden(encoder_out)
        decode_lengths = [l - 1 for l in lengths]
        T = max(decode_lengths)
        predictions = torch.zeros(B, T, self.vocab_size).to(encoder_out.device)
        alphas      = torch.zeros(B, T, encoder_out.size(1)).to(encoder_out.device)

        for t in range(T):
            bt = sum(l > t for l in decode_lengths)
            ctx, alpha = self.attention(encoder_out[:bt], h[:bt])
            gate = self.sigmoid(self.f_beta(h[:bt]))
            ctx  = gate * ctx
            h, c = self.lstm_cell(
                torch.cat([embeddings[:bt, t, :], ctx], dim=1),
                (h[:bt], c[:bt])
            )
            predictions[:bt, t, :] = self.fc(self.dropout(h))
            alphas[:bt, t, :]      = alpha

        return predictions, alphas, decode_lengths


# ══════════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data.clamp_(-grad_clip, grad_clip)


# ══════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("  IMAGE CAPTIONING — FLICKR30K TRAINING")
    print(f"  Device     : {device}")
    print(f"  Images dir : {IMAGES_DIR}")
    print(f"  CSV path   : {CSV_PATH}")
    print("="*60 + "\n")

    # Verify paths exist before starting
    if not os.path.exists(IMAGES_DIR):
        raise FileNotFoundError(
            f"\n Images folder not found:\n   {IMAGES_DIR}"
            f"\n   Check your USERNAME at the top of this file!"
        )
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"\n results.csv not found:\n   {CSV_PATH}"
            f"\n   Check your USERNAME at the top of this file!"
        )

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Vocabulary ─────────────────────────────────────────────────
    if os.path.exists(VOCAB_PATH):
        vocab = Vocabulary.load(VOCAB_PATH)
        print(f"  Vocabulary loaded: {len(vocab)} tokens\n")
    else:
        print("  Building vocabulary...")
        df = pd.read_csv(CSV_PATH, sep="|", engine="python")
        df.columns = [c.strip() for c in df.columns]
        if len(df.columns) == 3:
            df.columns = ["image_name", "comment_number", "comment"]
        vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
        vocab.build(df["comment"].dropna().astype(str).tolist())
        vocab.save(VOCAB_PATH)

    # ── Datasets ───────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = Flickr30kDataset(IMAGES_DIR, CSV_PATH, vocab, split="train")
    val_ds   = Flickr30kDataset(IMAGES_DIR, CSV_PATH, vocab, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}\n")

    # ── Models ─────────────────────────────────────────────────────
    encoder = EncoderCNN(embed_size=EMBED_SIZE).to(device)
    decoder = DecoderRNN(
        embed_size=EMBED_SIZE, decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),  encoder_dim=EMBED_SIZE,
        attention_dim=ATTENTION_DIM, dropout=DROPOUT,
    ).to(device)

    # ── Optimizers ─────────────────────────────────────────────────
    enc_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()), lr=ENCODER_LR
    )
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR)
    enc_sch  = ReduceLROnPlateau(enc_opt, patience=2)
    dec_sch  = ReduceLROnPlateau(dec_opt, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"]).to(device)

    best_val_loss = float("inf")
    print("Starting training...\n")

    for epoch in range(1, EPOCHS + 1):

        # ── Train ───────────────────────────────────────────────────
        encoder.train()
        decoder.train()
        train_loss = AverageMeter()
        t0 = time.time()

        for i, (imgs, caps, lengths) in enumerate(train_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)

            enc_opt.zero_grad()
            dec_opt.zero_grad()

            features = encoder(imgs)
            preds, alphas, dec_lens = decoder(features, caps, lengths)

            T   = max(dec_lens)
            out = preds[:, :T, :].reshape(-1, len(vocab))
            tgt = caps[:, 1:][:, :T].reshape(-1)

            loss  = criterion(out, tgt)
            loss += ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss.backward()

            clip_gradient(enc_opt, GRAD_CLIP)
            clip_gradient(dec_opt, GRAD_CLIP)
            enc_opt.step()
            dec_opt.step()

            train_loss.update(loss.item(), sum(dec_lens))

            if (i + 1) % PRINT_EVERY == 0:
                print(f"  Epoch [{epoch:>2}/{EPOCHS}] "
                      f"Step [{i+1:>4}/{len(train_loader)}] "
                      f"Loss: {train_loss.avg:.4f}  "
                      f"({time.time()-t0:.0f}s)")

        # ── Validate ────────────────────────────────────────────────
        encoder.eval()
        decoder.eval()
        val_loss = AverageMeter()

        with torch.no_grad():
            for imgs, caps, lengths in val_loader:
                imgs = imgs.to(device)
                caps = caps.to(device)
                features = encoder(imgs)
                preds, alphas, dec_lens = decoder(features, caps, lengths)
                T   = max(dec_lens)
                out = preds[:, :T, :].reshape(-1, len(vocab))
                tgt = caps[:, 1:][:, :T].reshape(-1)
                loss  = criterion(out, tgt)
                loss += ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
                val_loss.update(loss.item(), sum(dec_lens))

        enc_sch.step(val_loss.avg)
        dec_sch.step(val_loss.avg)

        print(f"\n{'─'*55}")
        print(f"  Epoch {epoch:>2} | Train: {train_loss.avg:.4f} | "
              f"Val: {val_loss.avg:.4f} | {time.time()-t0:.0f}s")
        print(f"{'─'*55}\n")

        # ── Save best checkpoint ────────────────────────────────────
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            ckpt_path = os.path.join(CKPT_DIR, "best_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "val_loss": best_val_loss,
                "vocab_size": len(vocab),
                "args": {
                    "embed_size": EMBED_SIZE,
                    "decoder_dim": DECODER_DIM,
                    "attention_dim": ATTENTION_DIM,
                    "dropout": DROPOUT,
                },
            }, ckpt_path)
            print(f"  ✔ Best model saved → {ckpt_path}\n")

    print(f"\n{'='*55}")
    print(f"   Training complete!")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {CKPT_DIR}\\best_checkpoint.pth")
    print(f"  Vocabulary    : {VOCAB_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()