# Forensic AI Investigation System — Setup Guide

A Flask-based forensic suspect-identification pipeline: witness audio → transcript → structured attributes → AI face generation → criminal database matching.

---

## 1. Requirements

- **Windows 10/11** (Linux/macOS also work, but `run.bat` is Windows)
- **Python 3.10+** — download from https://www.python.org/downloads/ (check "Add Python to PATH" during install).
- **~8 GB free disk space** (models download on first run)
- **Internet connection** for the initial download and for speech-to-text

---

## 2. First-time setup

1. **Unzip** the project to any folder, e.g. `C:\forensic-combined\`
2. **Double-click `run.bat`**

   On the first launch it will:
   - create a virtual environment (`venv\`)
   - install all Python packages
   - copy `.env.example` → `.env` and **stop**, asking you to add your API keys

3. **Edit `.env`** in Notepad and fill in:

   | Key | Required? | Where to get it |
   |---|---|---|
   | `SECRET_KEY` | Yes | Any long random string, e.g. 30 random characters |
   | `DEEPGRAM_API_KEY` | Yes (for audio) | Free signup at https://console.deepgram.com/ |
   | `HF_API_TOKEN` | Only for api mode | https://huggingface.co/settings/tokens |
   | `GENERATION_MODE` | Yes | `local` (recommended) or `api` |

4. **Double-click `run.bat` again**. It will:
   - start Flask at http://127.0.0.1:5000
   - download the AI models on the first generation request (one-time, ~4 GB)

---

## 3. Generation modes

### `GENERATION_MODE=local`  (recommended)
- Runs **SD 1.5 Realistic Vision + LCM-LoRA + ControlNet** on this machine
- First generation downloads ~4 GB of model weights, cached forever
- Fully offline after that — no API keys, no usage fees
- **Witness sketch uploads supported** (ControlNet sketch-to-photo)
- Speed: ~60 s per image on CPU, ~3 s on GPU

### `GENERATION_MODE=api`
- Uses **FLUX.1-schnell** via the HuggingFace Inference API
- No local downloads, ~10 s per image, ~$0.003 per image
- Requires a HuggingFace token **plus** a linked paid provider at
  https://huggingface.co/settings/inference-providers (fal-ai is cheapest)
- **No sketch upload support** (use local mode for sketches)

---

## 4. Default login

On the very first launch the system creates one admin account:

```
Username: superadmin
Password: Admin@123
```

**Change this password immediately after logging in.** The superadmin can then create Admin accounts, and Admins can create Police officers.

---

## 5. Roles

| Role | Can do |
|---|---|
| **Superadmin** | Manage admins, view audit logs |
| **Admin** | Manage police officers, manage criminal database |
| **Police** | Run the 4-step forensic pipeline, create cases |

---

## 6. The 4-step forensic pipeline (police role)

1. **Upload witness audio** → Deepgram Nova-2 transcribes it to text
2. **Review description** → the system parses it into structured attributes (gender, age, skin tone, marks, accessories, etc.) and builds a HiTS-style hierarchical prompt
3. **Generate candidate faces** → AI generates mugshots with accessories (bindi, earrings, nose ring) built-in, then scars/moles rendered at precise locations via landmark detection
4. **Match against criminal database** → ArcFace embedding (pure ONNX) + attribute matching finds the top suspects

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `run.bat` closes instantly | Open Command Prompt, `cd` into the folder, run `run.bat` there to see the error |
| "Python was not found" | Reinstall Python 3.10+ from python.org, tick **Add Python to PATH** |
| First generation takes 10+ minutes | Normal on first run (downloading ~4 GB). Subsequent runs are fast |
| `HuggingFace API returned 402` in api mode | Link a paid provider (fal-ai) at https://huggingface.co/settings/inference-providers, or switch to `GENERATION_MODE=local` |
| No faces detected by the matcher | Add criminal photos to `data/criminal_db/` as Admin, then click "Rebuild Database" |
| Port 5000 already in use | Close whatever else is using it, or edit `app.py` line with `app.run(..., port=5000)` |

---

## 8. Where things live

```
forensic-combined/
├── app.py                 # Flask app (all routes)
├── config.py              # Settings loader
├── .env                   # Your API keys (never share!)
├── modules/
│   ├── speech_to_text.py  # Deepgram integration
│   ├── attribute_parser.py # Witness-text → structured attributes
│   ├── face_generator.py  # Two-stage: AI generation + mark rendering
│   ├── face_enhancer.py   # Scar/mole rendering via face landmarks
│   ├── face_matcher.py    # ArcFace two-layer matching (pure ONNX)
│   └── onnx_face.py       # Face detection + recognition (no insightface)
├── database/              # SQL schema
├── data/
│   ├── forensic.db        # SQLite database (auto-created)
│   ├── criminal_db/       # Criminal face photos (added via Admin panel)
│   ├── generated_faces/   # AI-generated candidates (overwritten each run)
│   ├── uploads/           # Witness audio uploads
│   └── test_audio/        # Optional sample audio
├── templates/             # Jinja2 HTML templates
└── requirements.txt
```

---

## 9. Safety

Every AI generation request is wrapped with SFW guardrails:

- Positive prompt always includes `"wearing a plain gray collared shirt, clothed upper body, professional mugshot photograph"`
- Negative prompt blocks `nude, naked, nudity, nsfw, topless, shirtless, bare chest, ...`

These apply to both local and api modes.

---

## 10. Support

If something breaks:

1. Check the `run.bat` console window for Python errors
2. Check `data/forensic.db` hasn't been corrupted (delete it — it'll recreate on next start)
3. If a generation fails, check the console for the specific error message (402 = need paid provider, 401 = bad token, etc.)
