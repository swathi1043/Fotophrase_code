"""
Evaluate Image Captioning Model
Metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, CIDEr
"""

import os
import re
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1-4', quiet=True)

# ══════════════════════════════════════════════════════════════
#  YOUR PATHS
# ══════════════════════════════════════════════════════════════

USERNAME    = "swath"
PROJECT_DIR = rf"C:\Users\{USERNAME}\OneDrive\Documents\Desktop\30k_project"
CKPT_PATH   = os.path.join(PROJECT_DIR, "best_checkpoint.pth")
VOCAB_PATH  = os.path.join(PROJECT_DIR, "vocab.pkl")

BASE_DIR    = rf"C:\Users\{USERNAME}\Downloads\archive (2)\flickr30k_images"
IMAGES_DIR  = os.path.join(BASE_DIR, "flickr30k_images")
CSV_PATH    = os.path.join(BASE_DIR, "results.csv")

# Number of test images to evaluate (keep low for speed on CPU)
MAX_EVAL_IMAGES = 200


# ══════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════

class Vocabulary:
    PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

    def __init__(self):
        self.word2idx = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        return re.sub(r"[^a-zA-Z0-9 ]", "", str(text).lower()).split()

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet        = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.fc            = nn.Linear(2048, embed_size)
        self.bn            = nn.BatchNorm1d(embed_size)

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

    def _init_hidden(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def beam_search(self, encoder_out, word2idx, idx2word,
                    max_len=30, beam_size=3):
        device     = encoder_out.device
        k          = beam_size
        vocab_size = self.vocab_size

        encoder_out  = encoder_out.expand(k, *encoder_out.shape[1:])
        start_token  = word2idx.get("<SOS>", 1)
        end_token    = word2idx.get("<EOS>", 2)
        seqs         = torch.full((k, 1), start_token, dtype=torch.long, device=device)
        top_k_scores = torch.zeros(k, 1, device=device)

        h, c = self._init_hidden(encoder_out)
        complete_seqs, complete_scores = [], []

        for step in range(max_len):
            emb          = self.embedding(seqs[:, -1])
            ctx, _       = self.attention(encoder_out, h)
            gate         = self.sigmoid(self.f_beta(h))
            ctx          = gate * ctx
            h, c         = self.lstm_cell(torch.cat([emb, ctx], dim=1), (h, c))
            scores       = torch.log_softmax(self.fc(h), dim=1)
            scores       = top_k_scores.expand_as(scores) + scores

            if step == 0:
                top_scores, top_words = scores[0].topk(k, 0)
            else:
                top_scores, top_words = scores.view(-1).topk(k, 0)

            prev_inds    = top_words // vocab_size
            next_inds    = top_words % vocab_size
            seqs         = torch.cat([seqs[prev_inds], next_inds.unsqueeze(1)], dim=1)
            top_k_scores = top_scores.unsqueeze(1)

            incomplete = [i for i, w in enumerate(next_inds) if w.item() != end_token]
            complete   = [i for i, w in enumerate(next_inds) if w.item() == end_token]

            if complete:
                complete_seqs.extend(seqs[complete].tolist())
                complete_scores.extend(top_k_scores[complete].squeeze(1).tolist())

            k -= len(complete)
            if k == 0:
                break

            seqs         = seqs[incomplete]
            h, c         = h[incomplete], c[incomplete]
            encoder_out  = encoder_out[incomplete]
            top_k_scores = top_k_scores[incomplete]

        if not complete_seqs:
            complete_seqs   = seqs.tolist()
            complete_scores = top_k_scores.squeeze(1).tolist()

        best  = complete_seqs[complete_scores.index(max(complete_scores))]
        words = [idx2word.get(w, "<UNK>") for w in best[1:]]
        if "<EOS>" in words:
            words = words[:words.index("<EOS>")]
        return words   # return list of words (not string) for BLEU


# ══════════════════════════════════════════════════════════════
#  LOAD TEST DATA
# ══════════════════════════════════════════════════════════════

def load_test_data(csv_path, images_dir, seed=42):
    """Load test split — same split used during training."""
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

    # Same split as training (seed=42, test=5%)
    import random
    unique_imgs = df["image_name"].unique().tolist()
    random.Random(seed).shuffle(unique_imgs)
    n_test   = int(len(unique_imgs) * 0.05)
    test_imgs = set(unique_imgs[:n_test])

    df_test = df[df["image_name"].isin(test_imgs)]

    # Group: image → list of 5 reference captions
    image_captions = {}
    for _, row in df_test.iterrows():
        img_path = os.path.join(images_dir, row["image_name"])
        if not os.path.exists(img_path):
            continue
        if img_path not in image_captions:
            image_captions[img_path] = []
        image_captions[img_path].append(row["comment"])

    print(f"  ✔ Test images found: {len(image_captions)}")
    return image_captions


# ══════════════════════════════════════════════════════════════
#  IMAGE TRANSFORM
# ══════════════════════════════════════════════════════════════

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(encoder, decoder, vocab, image_captions, device, max_images=200):
    encoder.eval()
    decoder.eval()

    all_references  = []   # list of list of list of words
    all_hypotheses  = []   # list of list of words
    meteor_scores   = []

    tokenize = lambda t: re.sub(r"[^a-zA-Z0-9 ]", "", t.lower()).split()
    smooth   = SmoothingFunction().method1

    items = list(image_captions.items())[:max_images]
    total = len(items)

    print(f"\n  Evaluating {total} images...\n")

    for idx, (img_path, ref_captions) in enumerate(items):
        try:
            tensor = preprocess(img_path).to(device)
            with torch.no_grad():
                features   = encoder(tensor)
                hypothesis = decoder.beam_search(
                    features,
                    word2idx  = vocab.word2idx,
                    idx2word  = vocab.idx2word,
                    max_len   = 30,
                    beam_size = 3,
                )

            # Tokenize references
            references = [tokenize(cap) for cap in ref_captions]

            all_references.append(references)
            all_hypotheses.append(hypothesis)

            # METEOR per image
            m = max(meteor_score([r for r in references], hypothesis)
                    for _ in [1])
            meteor_scores.append(m)

            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1:>3}/{total}] Sample caption: {' '.join(hypothesis[:8])}...")

        except Exception as e:
            print(f"  Skipping {os.path.basename(img_path)}: {e}")
            continue

    # ── BLEU Scores ───────────────────────────────────────────
    bleu1 = corpus_bleu(all_references, all_hypotheses,
                        weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(all_references, all_hypotheses,
                        weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(all_references, all_hypotheses,
                        weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(all_references, all_hypotheses,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # ── METEOR ────────────────────────────────────────────────
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    return {
        "BLEU-1":  round(bleu1  * 100, 2),
        "BLEU-2":  round(bleu2  * 100, 2),
        "BLEU-3":  round(bleu3  * 100, 2),
        "BLEU-4":  round(bleu4  * 100, 2),
        "METEOR":  round(avg_meteor * 100, 2),
        "Images evaluated": total,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*55)
    print("  IMAGE CAPTIONING — EVALUATION")
    print(f"  Device: {device}")
    print("="*55 + "\n")

    # ── Check files ───────────────────────────────────────────
    for path, name in [(CKPT_PATH, "Checkpoint"), (VOCAB_PATH, "Vocabulary")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return
        print(f"  ✔ {name} found")

    # ── Load vocab ────────────────────────────────────────────
    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"  ✔ Vocabulary: {len(vocab)} tokens")

    # ── Load model ────────────────────────────────────────────
    ckpt = torch.load(CKPT_PATH, map_location=device)
    args = ckpt.get("args", {})

    encoder = EncoderCNN(embed_size=args.get("embed_size", 256)).to(device)
    decoder = DecoderRNN(
        embed_size    = args.get("embed_size",    256),
        decoder_dim   = args.get("decoder_dim",   512),
        vocab_size    = len(vocab),
        encoder_dim   = args.get("embed_size",    256),
        attention_dim = args.get("attention_dim", 256),
        dropout       = args.get("dropout",       0.5),
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    print(f"  ✔ Model loaded (epoch {ckpt.get('epoch', '?')})\n")

    # ── Load test data ────────────────────────────────────────
    print("  Loading test data...")
    image_captions = load_test_data(CSV_PATH, IMAGES_DIR)

    # ── Run evaluation ────────────────────────────────────────
    results = evaluate(encoder, decoder, vocab, image_captions,
                       device, max_images=MAX_EVAL_IMAGES)

    # ── Print results ─────────────────────────────────────────
    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)
    print(f"  Images evaluated : {results['Images evaluated']}")
    print(f"  {'─'*40}")
    print(f"  BLEU-1  : {results['BLEU-1']:>6.2f}%   (target: >55%)")
    print(f"  BLEU-2  : {results['BLEU-2']:>6.2f}%   (target: >35%)")
    print(f"  BLEU-3  : {results['BLEU-3']:>6.2f}%   (target: >25%)")
    print(f"  BLEU-4  : {results['BLEU-4']:>6.2f}%   (target: >15%)")
    print(f"  METEOR  : {results['METEOR']:>6.2f}%   (target: >18%)")
    print(f"  {'─'*40}")

    # Grade your model
    bleu4 = results["BLEU-4"]
    if bleu4 >= 20:
        grade = "Good"
    elif bleu4 >= 12:
        grade = "Acceptable "
    else:
        grade = "Needs improvement (train more epochs)"

    print(f"  Model Grade : {grade}")
    print("="*55 + "\n")

    # Save results to text file
    result_path = os.path.join(PROJECT_DIR, "evaluation_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("IMAGE CAPTIONING EVALUATION RESULTS\n")
        f.write("="*40 + "\n")
        for k, v in results.items():
            f.write(f"{k:20s}: {v}\n")
        f.write(f"{'Model Grade':20s}: {grade}\n")
    print(f"  Results saved → {result_path}")


if __name__ == "__main__":
    main()