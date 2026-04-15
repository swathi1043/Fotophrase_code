"""
FotoPhrase — Image Caption Generator with Voice
Model: ResNet50 + Attention LSTM (your trained model)
"""

import os
import re
import pickle
import threading
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
from PIL import Image
import customtkinter as ctk

# ══════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════

USERNAME    = "swath"
PROJECT_DIR = rf"C:\Users\{USERNAME}\OneDrive\Documents\Desktop\30k_project"
CKPT_PATH   = os.path.join(PROJECT_DIR, "best_checkpoint.pth")
VOCAB_PATH  = os.path.join(PROJECT_DIR, "vocab.pkl")
LOG_PATH    = os.path.join(PROJECT_DIR, "captions_log.txt")

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

    def beam_search(self, encoder_out, word2idx, idx2word, max_len=30, beam_size=3):
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
        return " ".join(words)


# ══════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess(pil_image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(pil_image.convert("RGB")).unsqueeze(0)


# ══════════════════════════════════════════════════════════════
#  VOICE
# ══════════════════════════════════════════════════════════════

def speak_text(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass

def assistant_speak(text):
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════

class FotoPhrase(ctk.CTk):
    def __init__(self, encoder, decoder, vocab, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab   = vocab
        self.device  = device

        self.current_image = None
        self.cam_active    = False
        self.cap           = None

        # ── Theme ─────────────────────────────────────────────
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("FOTOPHRASE  |  The Smart AI Writer & Reader")
        self.geometry("1050x660")
        self.resizable(False, False)

        # ── Grid ──────────────────────────────────────────────
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()
        self._build_history()

        # Status bar
        self.status_label = ctk.CTkLabel(
            self, text="System Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_label.grid(row=1, column=1, sticky="w", padx=30, pady=6)

        # Welcome voice
        self.after(1000, lambda: assistant_speak(
            "Welcome to FotoPhrase. Click Upload Gallery or Open Camera to begin."
        ))

    # ── Sidebar ───────────────────────────────────────────────

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(
            self.sidebar, text="FotoPhrase",
            font=ctk.CTkFont(size=26, weight="bold")
        ).pack(pady=(30, 4))

        ctk.CTkLabel(
            self.sidebar, text="AI Caption Generator",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 24))

        self.btn_gallery = ctk.CTkButton(
            self.sidebar, text="open Gallery",
            command=self.load_from_gallery,
            height=38, font=ctk.CTkFont(size=13)
        )
        self.btn_gallery.pack(pady=8, padx=20, fill="x")

        self.btn_camera = ctk.CTkButton(
            self.sidebar, text="Open Camera",
            command=self.load_from_camera,
            height=38, font=ctk.CTkFont(size=13),
            fg_color="#2ecc71", hover_color="#27ae60"
        )
        self.btn_camera.pack(pady=8, padx=20, fill="x")

        self.btn_capture = ctk.CTkButton(
            self.sidebar, text="📸  Capture Photo",
            command=self.capture_photo,
            height=38, font=ctk.CTkFont(size=13),
            fg_color="#e74c3c", hover_color="#c0392b"
        )
        self.btn_capture.pack(pady=8, padx=20, fill="x")

        self.btn_generate = ctk.CTkButton(
            self.sidebar, text="✨  Generate Caption",
            command=self.generate_caption,
            height=42, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#9b59b6", hover_color="#8e44ad"
        )
        self.btn_generate.pack(pady=(16, 8), padx=20, fill="x")

        self.btn_save = ctk.CTkButton(
            self.sidebar, text="💾  Save Caption",
            command=self.save_to_file,
            height=36, font=ctk.CTkFont(size=12),
            fg_color="transparent", border_width=2
        )
        self.btn_save.pack(pady=8, padx=20, fill="x")

        # Model info at bottom
        ctk.CTkLabel(
            self.sidebar,
            text="Model: ResNet50 + LSTM\nDataset: Flickr30k\nVocab: 7,727 tokens",
            font=ctk.CTkFont(size=10),
            text_color="gray", justify="left"
        ).pack(side="bottom", pady=20, padx=16, anchor="w")

    # ── Main Panel ────────────────────────────────────────────

    def _build_main(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.image_label = ctk.CTkLabel(
            self.main_frame,
            text="Select an image to begin\n\nUse Gallery or Camera from the sidebar",
            width=580, height=340,
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.image_label.pack(pady=(20, 10), padx=20)

        ctk.CTkLabel(
            self.main_frame, text="Generated Caption:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=24)

        self.caption_box = ctk.CTkTextbox(
            self.main_frame, width=560, height=90,
            font=ctk.CTkFont(size=16),
            corner_radius=10
        )
        self.caption_box.pack(pady=(4, 16), padx=20)
        self.caption_box.insert("0.0", "AI description will appear here...")
        self.caption_box.configure(state="disabled")

    # ── History Panel ─────────────────────────────────────────

    def _build_history(self):
        self.history_frame = ctk.CTkFrame(self, width=240, corner_radius=15)
        self.history_frame.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        self.history_frame.grid_propagate(False)

        ctk.CTkLabel(
            self.history_frame, text="Session History",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(16, 6))

        ctk.CTkLabel(
            self.history_frame, text="All captions this session",
            font=ctk.CTkFont(size=10), text_color="gray"
        ).pack(pady=(0, 8))

        self.history_list = ctk.CTkTextbox(
            self.history_frame, width=210, height=480,
            font=ctk.CTkFont(size=11),
            corner_radius=8
        )
        self.history_list.pack(pady=6, padx=14)
        self.history_list.configure(state="disabled")

    # ── Gallery ───────────────────────────────────────────────

    def load_from_gallery(self):
        self._stop_webcam()
        assistant_speak("Opening gallery. Please select an image.")
        path = ctk.filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if path:
            self.current_image = Image.open(path).convert("RGB")
            self._show_preview(self.current_image)
            self._set_status(f"Loaded: {os.path.basename(path)}")
            self._set_caption("Image loaded. Click Generate Caption.")

    # ── Camera ────────────────────────────────────────────────

    def load_from_camera(self):
        try:
            import cv2
        except ImportError:
            self._set_status("OpenCV not installed. Run: pip install opencv-python")
            return

        if self.cam_active:
            self._set_status("Webcam already open — click Capture Photo")
            return

        import cv2
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("Could not open webcam.")
            return

        self.cam_active = True
        assistant_speak("Camera is open. Click Capture Photo when ready.")
        self._set_status("Webcam live — click Capture Photo")
        self._update_webcam_feed()

    def _update_webcam_feed(self):
        if not self.cam_active or self.cap is None:
            return
        import cv2
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)
            self._show_preview(pil_img, store=False)
            self._live_frame = pil_img
        self.after(30, self._update_webcam_feed)

    def capture_photo(self):
        if not self.cam_active:
            self._set_status("Open Camera first, then capture.")
            return
        if hasattr(self, "_live_frame"):
            self.current_image = self._live_frame.copy()
            self._show_preview(self.current_image)
            self._stop_webcam()
            assistant_speak("Photo captured. Click Generate Caption.")
            self._set_status("Photo captured from webcam")
            self._set_caption("Image loaded. Click Generate Caption.")

    def _stop_webcam(self):
        self.cam_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ── Preview ───────────────────────────────────────────────

    def _show_preview(self, pil_img, store=True):
        preview = pil_img.copy()
        preview.thumbnail((560, 320))
        ctk_img = ctk.CTkImage(
            light_image=preview, dark_image=preview,
            size=(min(preview.width, 560), min(preview.height, 320))
        )
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    # ── Caption Generation ────────────────────────────────────

    def generate_caption(self):
        if self.current_image is None:
            self._set_status("Please select or capture an image first.")
            return

        self._set_status("AI is analyzing image...")
        self._set_caption("Please wait...")
        assistant_speak("Please wait. I am analyzing the image.")

        self.btn_gallery.configure(state="disabled")
        self.btn_camera.configure(state="disabled")
        self.btn_generate.configure(state="disabled")

        threading.Thread(target=self._run_caption, daemon=True).start()

    def _run_caption(self):
        try:
            tensor = preprocess(self.current_image).to(self.device)
            with torch.no_grad():
                features = self.encoder(tensor)
                caption  = self.decoder.beam_search(
                    features,
                    word2idx  = self.vocab.word2idx,
                    idx2word  = self.vocab.idx2word,
                    max_len   = 30,
                    beam_size = 3,
                )
            caption = caption.capitalize() + "."
            self.after(0, lambda: self._on_caption_done(caption))
        except Exception as e:
            self.after(0, lambda: self._set_status(f"Error: {e}"))
            self.after(0, self._enable_buttons)

    def _on_caption_done(self, caption):
        self._set_caption(caption)
        self._set_status("Analysis complete.")
        self._add_to_history(caption)
        self._enable_buttons()
        assistant_speak(caption)

    def _enable_buttons(self):
        self.btn_gallery.configure(state="normal")
        self.btn_camera.configure(state="normal")
        self.btn_generate.configure(state="normal")

    # ── Save Caption ──────────────────────────────────────────

    def save_to_file(self):
        text = self.caption_box.get("0.0", "end").strip()
        if text and "AI description" not in text and "Please wait" not in text:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
            self._set_status(f"Caption saved to captions_log.txt")
            assistant_speak("Caption saved successfully.")
        else:
            self._set_status("No caption to save yet.")

    # ── History ───────────────────────────────────────────────

    def _add_to_history(self, caption):
        timestamp = datetime.now().strftime("%H:%M")
        self.history_list.configure(state="normal")
        self.history_list.insert("0.0", f"[{timestamp}]\n{caption}\n\n")
        self.history_list.configure(state="disabled")

    # ── Helpers ───────────────────────────────────────────────

    def _set_status(self, text):
        self.status_label.configure(text=text)

    def _set_caption(self, text):
        self.caption_box.configure(state="normal")
        self.caption_box.delete("0.0", "end")
        self.caption_box.insert("0.0", text)
        self.caption_box.configure(state="disabled")

    def on_close(self):
        self._stop_webcam()
        self.destroy()


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL & LAUNCH
# ══════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*55)
    print("  FOTOPHRASE — LOADING MODEL")
    print("="*55)

    for path, name in [(CKPT_PATH, "best_checkpoint.pth"), (VOCAB_PATH, "vocab.pkl")]:
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            input("Press Enter to exit...")
            return
        print(f"  Found: {name}")

    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"  Vocabulary: {len(vocab)} tokens")

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
    encoder.eval()
    decoder.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')})")
    print("\n  Launching FotoPhrase GUI...\n")

    app = FotoPhrase(encoder, decoder, vocab, device)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()