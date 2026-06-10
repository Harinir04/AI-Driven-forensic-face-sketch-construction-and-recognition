"""
Generate Setup Guide PDF for the Forensic AI Investigation System.
Run: python generate_setup_pdf.py
Output: Setup_Guide.pdf in the project root.
"""

from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, ListFlowable, ListItem
    )
except ImportError:
    print("Installing reportlab...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, ListFlowable, ListItem
    )


# ── Colors ──
DARK_BLUE = HexColor("#1a237e")
MID_BLUE = HexColor("#283593")
LIGHT_BLUE = HexColor("#e8eaf6")
ACCENT = HexColor("#ff6f00")
DARK_GRAY = HexColor("#333333")
LIGHT_GRAY = HexColor("#f5f5f5")
WHITE = HexColor("#ffffff")
BLACK = HexColor("#000000")
GREEN = HexColor("#2e7d32")
RED = HexColor("#c62828")


def build_pdf():
    output = Path(__file__).parent / "Setup_Guide.pdf"

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=28, leading=34, textColor=DARK_BLUE,
        spaceAfter=6 * mm, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "CoverSub", parent=styles["Normal"],
        fontSize=14, leading=18, textColor=MID_BLUE,
        spaceAfter=4 * mm, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=18, leading=22, textColor=DARK_BLUE,
        spaceBefore=8 * mm, spaceAfter=4 * mm,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=14, leading=17, textColor=MID_BLUE,
        spaceBefore=5 * mm, spaceAfter=3 * mm,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=11, leading=15, textColor=DARK_GRAY,
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=HexColor("#1b5e20"),
        fontName="Courier", backColor=LIGHT_GRAY,
        leftIndent=8 * mm, spaceAfter=2 * mm,
        borderPadding=(2 * mm, 2 * mm, 2 * mm, 2 * mm),
    ))
    styles.add(ParagraphStyle(
        "Warning", parent=styles["Normal"],
        fontSize=11, leading=15, textColor=RED,
        fontName="Helvetica-Bold", spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "Success", parent=styles["Normal"],
        fontSize=11, leading=15, textColor=GREEN,
        fontName="Helvetica-Bold", spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        "StepNum", parent=styles["Normal"],
        fontSize=13, leading=16, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=MID_BLUE,
        leftIndent=5 * mm, spaceAfter=2 * mm,
        fontName="Helvetica-Oblique",
    ))

    story = []

    def hr():
        story.append(Spacer(1, 2 * mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BLUE))
        story.append(Spacer(1, 2 * mm))

    def code(text):
        story.append(Paragraph(text, styles["CodeBlock"]))

    def body(text):
        story.append(Paragraph(text, styles["Body"]))

    def note(text):
        story.append(Paragraph(text, styles["Note"]))

    def bullet_list(items):
        for item in items:
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{item}", styles["Body"]))

    def step_table(steps):
        """Create a numbered step table with blue circles."""
        for i, (title, desc) in enumerate(steps, 1):
            data = [[
                Paragraph(f"<b>{i}</b>", styles["StepNum"]),
                Paragraph(f"<b>{title}</b><br/>{desc}", styles["Body"]),
            ]]
            t = Table(data, colWidths=[12 * mm, 145 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (0, 0), WHITE),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROUNDEDCORNERS", [3, 3, 3, 3]),
                ("LEFTPADDING", (1, 0), (1, 0), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]))
            story.append(t)
            story.append(Spacer(1, 3 * mm))

    # ══════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════

    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph("Forensic AI Investigation System", styles["CoverTitle"]))
    story.append(Paragraph("Setup &amp; Installation Guide", styles["CoverSub"]))
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="60%", thickness=2, color=DARK_BLUE))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "Witness Audio &rarr; Transcript &rarr; Facial Attributes &rarr; "
        "AI Face Generation &rarr; Criminal Database Matching",
        styles["CoverSub"]
    ))
    story.append(Spacer(1, 20 * mm))

    info_data = [
        ["Version", "2.0 (Pure ONNX)"],
        ["Python", "3.10, 3.11, 3.12, 3.13, 3.14"],
        ["Platform", "Windows 10/11 (Linux/macOS compatible)"],
        ["Install Size", "~50 MB (API mode) / ~2.5 GB (Local mode)"],
    ]
    info_table = Table(info_data, colWidths=[40 * mm, 120 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BLUE),
    ]))
    story.append(info_table)

    story.append(PageBreak())

    # ══════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════

    story.append(Paragraph("Table of Contents", styles["H1"]))
    hr()
    toc_items = [
        "1. Requirements",
        "2. Quick Start (5 Minutes)",
        "3. Generation Modes",
        "4. Default Login &amp; Roles",
        "5. The 4-Step Forensic Pipeline",
        "6. Project Structure",
        "7. Troubleshooting",
        "8. Appendix: Manual Setup",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["Body"]))
    story.append(Spacer(1, 5 * mm))

    story.append(PageBreak())

    # ══════════════════════════════════════════
    # 1. REQUIREMENTS
    # ══════════════════════════════════════════

    story.append(Paragraph("1. Requirements", styles["H1"]))
    hr()

    req_data = [
        ["Requirement", "Details"],
        ["Operating System", "Windows 10 or 11 (Linux/macOS also work)"],
        ["Python", "3.10 or higher (3.12 recommended)"],
        ["Disk Space", "~50 MB for API mode, ~2.5 GB for Local mode"],
        ["Internet", "Required for first setup and speech-to-text"],
        ["RAM", "4 GB minimum, 8 GB recommended for local mode"],
    ]
    req_table = Table(req_data, colWidths=[40 * mm, 120 * mm])
    req_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    story.append(req_table)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Install Python", styles["H2"]))
    body("If Python is not installed, download from:")
    code("https://www.python.org/downloads/")
    story.append(Paragraph(
        '<b><font color="#c62828">IMPORTANT:</font></b> '
        'During installation, CHECK the box "Add Python to PATH".',
        styles["Warning"]
    ))

    # ══════════════════════════════════════════
    # 2. QUICK START
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("2. Quick Start (5 Minutes)", styles["H1"]))
    hr()

    step_table([
        ("Extract the project",
         "Unzip the project folder to any location.<br/>"
         'Example: <font face="Courier" color="#1b5e20">C:\\forensic-ai\\</font>'),

        ("Double-click run.bat",
         "The script will:<br/>"
         "&bull; Detect your Python version automatically<br/>"
         "&bull; Create a virtual environment<br/>"
         "&bull; Create a .env file from template<br/>"
         "&bull; Stop and ask you to fill in API keys"),

        ("Edit the .env file",
         "Open .env in Notepad and fill in your API keys (see table below).<br/>"
         "Save the file."),

        ("Double-click run.bat again",
         "The script will install all dependencies and start the server.<br/>"
         'You will see: <font face="Courier" color="#1b5e20">Running on http://127.0.0.1:5000</font>'),

        ("Open your browser",
         'Go to <font face="Courier" color="#1b5e20">http://127.0.0.1:5000</font><br/>'
         'Login with: <b>superadmin</b> / <b>Admin@123</b><br/>'
         '<font color="#c62828">Change this password immediately after first login.</font>'),
    ])

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("API Keys for .env file:", styles["H2"]))

    env_data = [
        ["Key", "Required?", "Where to get it"],
        ["SECRET_KEY", "Yes", "Any long random string (30+ characters)"],
        ["DEEPGRAM_API_KEY", "Yes (for audio)", "Free at https://console.deepgram.com/"],
        ["HF_API_TOKEN", "Only for API mode", "https://huggingface.co/settings/tokens"],
        ["GENERATION_MODE", "Yes", '"api" (lightweight) or "local" (offline)'],
    ]
    env_table = Table(env_data, colWidths=[38 * mm, 28 * mm, 94 * mm])
    env_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    story.append(env_table)

    # ══════════════════════════════════════════
    # 3. GENERATION MODES
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("3. Generation Modes", styles["H1"]))
    hr()

    story.append(Paragraph("API Mode (Recommended for quick setup)", styles["H2"]))
    body("Set <b>GENERATION_MODE=api</b> in .env")
    bullet_list([
        "Uses FLUX.1-schnell via HuggingFace cloud API",
        "Lightweight install: only ~50 MB of packages",
        "Fast: ~10 seconds per face image",
        "Cost: ~$0.003 per image (very cheap)",
        "Requires: HF_API_TOKEN + linked paid provider (fal-ai recommended)",
        "No sketch upload support",
    ])
    story.append(Spacer(1, 3 * mm))
    note("To link a paid provider: go to https://huggingface.co/settings/inference-providers and link fal-ai.")

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Local Mode (Recommended for offline/air-gapped)", styles["H2"]))
    body("Set <b>GENERATION_MODE=local</b> in .env")
    bullet_list([
        "Runs Stable Diffusion 1.5 on your machine",
        "First run downloads ~2.5 GB of model weights (one-time)",
        "Fully offline after first download",
        "Speed: ~3 minutes per image on CPU, ~15 seconds on GPU",
        "Supports witness sketch uploads (ControlNet)",
        "No API keys needed for generation (Deepgram still needs key for audio)",
    ])

    story.append(Spacer(1, 5 * mm))
    mode_cmp = [
        ["", "API Mode", "Local Mode"],
        ["Install size", "~50 MB", "~2.5 GB"],
        ["Speed per image", "~10 seconds", "~3 min (CPU)"],
        ["Internet needed", "Always", "First run only"],
        ["Cost", "~$0.003/image", "Free"],
        ["Sketch support", "No", "Yes"],
        ["Best for", "Quick setup", "Offline use"],
    ]
    mode_table = Table(mode_cmp, colWidths=[38 * mm, 60 * mm, 60 * mm])
    mode_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (0, -1), LIGHT_BLUE),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    story.append(mode_table)

    # ══════════════════════════════════════════
    # 4. LOGIN & ROLES
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("4. Default Login &amp; Roles", styles["H1"]))
    hr()

    story.append(Paragraph("First Login", styles["H2"]))
    login_data = [
        ["Username", "superadmin"],
        ["Password", "Admin@123"],
    ]
    login_table = Table(login_data, colWidths=[40 * mm, 80 * mm])
    login_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Courier-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("TEXTCOLOR", (0, 0), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    story.append(login_table)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        '<font color="#c62828"><b>Change this password immediately after first login.</b></font>',
        styles["Body"]
    ))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("User Roles", styles["H2"]))
    roles_data = [
        ["Role", "Capabilities"],
        ["Superadmin", "Create/manage Admin accounts, view audit logs"],
        ["Admin", "Create Police officers, manage criminal database, rebuild face index"],
        ["Police", "Run the 4-step forensic pipeline, create investigation cases"],
    ]
    roles_table = Table(roles_data, colWidths=[30 * mm, 130 * mm])
    roles_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    story.append(roles_table)

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Role Hierarchy", styles["H2"]))
    body("Superadmin &rarr; creates Admins &rarr; Admins create Police officers &rarr; Police run investigations")

    # ══════════════════════════════════════════
    # 5. THE 4-STEP PIPELINE
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("5. The 4-Step Forensic Pipeline", styles["H1"]))
    hr()
    body("Login as a <b>Police</b> user and click <b>New Case</b> to start.")
    story.append(Spacer(1, 3 * mm))

    step_table([
        ("Upload Witness Audio",
         "Record or upload a witness audio file (WAV, MP3, M4A, OGG, WEBM).<br/>"
         "Deepgram Nova-2 transcribes it to text automatically.<br/>"
         "Supports multiple languages with auto-detection."),

        ("Review Parsed Description",
         "The system extracts facial attributes from the transcript:<br/>"
         "&bull; Gender, age, ethnicity, skin tone<br/>"
         "&bull; Hair color/style, eye color, facial hair<br/>"
         "&bull; Face shape, nose type, build<br/>"
         "&bull; Scars, moles, birthmarks (with precise location)<br/>"
         "&bull; Accessories: bindi, earrings, nose ring<br/>"
         "You can edit the text and re-parse if needed."),

        ("Generate Candidate Faces",
         "The AI generates 4 face images based on the description.<br/>"
         "<b>Stage 1:</b> AI model creates the base face with accessories.<br/>"
         "<b>Stage 2:</b> Scars and moles are rendered at precise locations "
         "using face landmark detection.<br/>"
         "You can optionally upload a witness sketch for guided generation (local mode only)."),

        ("Match Against Criminal Database",
         "Select the best matching generated face.<br/>"
         "The system compares it against the criminal database using:<br/>"
         "&bull; <b>ArcFace</b> structural similarity (face shape, proportions)<br/>"
         "&bull; <b>Attribute matching</b> (scars, gender, age, skin tone)<br/>"
         "Top 5 matches are shown with confidence scores."),
    ])

    note("Before matching works, an Admin must add criminal records with face photos "
         "via the Admin panel and click 'Rebuild Face Database'.")

    # ══════════════════════════════════════════
    # 6. PROJECT STRUCTURE
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("6. Project Structure", styles["H1"]))
    hr()

    files = [
        ["File / Folder", "Purpose"],
        ["run.bat", "Double-click to start (auto-detects Python, installs deps)"],
        ["setup_check.py", "Diagnostic script: python setup_check.py"],
        ["app.py", "Flask web application (all routes)"],
        ["config.py", "Configuration loader (reads .env)"],
        [".env", "Your API keys (never share this file!)"],
        [".env.example", "Template for .env"],
        ["requirements.txt", "Core Python packages (~50 MB)"],
        ["requirements-local.txt", "Additional packages for local mode (~2.5 GB)"],
        ["modules/speech_to_text.py", "Deepgram audio transcription"],
        ["modules/attribute_parser.py", "Witness text to structured attributes"],
        ["modules/face_generator.py", "Two-stage AI face generation"],
        ["modules/face_enhancer.py", "Scar/mole rendering via landmarks"],
        ["modules/face_matcher.py", "ArcFace two-layer matching"],
        ["modules/onnx_face.py", "Face detection + recognition (pure ONNX)"],
        ["data/criminal_db/", "Criminal face photos (added via Admin panel)"],
        ["data/generated_faces/", "AI-generated faces (overwritten each run)"],
        ["data/forensic.db", "SQLite database (auto-created)"],
        ["database/schema.sql", "Database schema"],
        ["templates/", "HTML templates for web UI"],
    ]
    file_table = Table(files, colWidths=[52 * mm, 108 * mm])
    file_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * mm),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
    ]))
    story.append(file_table)

    # ══════════════════════════════════════════
    # 7. TROUBLESHOOTING
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("7. Troubleshooting", styles["H1"]))
    hr()

    issues = [
        ["Symptom", "Solution"],
        ["run.bat closes instantly",
         "Open Command Prompt, cd into the project folder, type run.bat to see the error"],
        ['"Python was not found"',
         "Install Python 3.10+ from python.org. CHECK 'Add Python to PATH'"],
        ["First generation is very slow",
         "Normal for local mode on CPU (~3 min/image). Use API mode for faster results"],
        ["HuggingFace 402 error",
         "Link a paid provider at huggingface.co/settings/inference-providers "
         "OR switch to GENERATION_MODE=local"],
        ["No matches found",
         "Criminal database is empty. Add criminals with photos via Admin panel, "
         "then click 'Rebuild Face Database'"],
        ["Port 5000 in use",
         "Close the other application using port 5000, or edit app.py to change the port"],
        ["Module import errors",
         "Run: python setup_check.py --install"],
        ["Database corrupted",
         "Delete data/forensic.db and restart. It recreates automatically"],
    ]
    issue_table = Table(issues, colWidths=[42 * mm, 118 * mm])
    issue_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, WHITE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(issue_table)

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Diagnostic Command", styles["H2"]))
    body("Run this to check your entire setup:")
    code("python setup_check.py")
    body("To auto-install missing packages:")
    code("python setup_check.py --install")

    # ══════════════════════════════════════════
    # 8. APPENDIX: MANUAL SETUP
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(Paragraph("8. Appendix: Manual Setup", styles["H1"]))
    hr()
    body("If run.bat does not work, follow these steps manually in Command Prompt:")

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Step 1: Open Command Prompt", styles["H2"]))
    code("Win+R &rarr; type cmd &rarr; press Enter")

    story.append(Paragraph("Step 2: Navigate to project folder", styles["H2"]))
    code('cd "C:\\path\\to\\forensic-face-generation"')

    story.append(Paragraph("Step 3: Check Python version", styles["H2"]))
    code("python --version")
    body("Should show Python 3.10 or higher.")

    story.append(Paragraph("Step 4: Create virtual environment", styles["H2"]))
    code("python -m venv venv")

    story.append(Paragraph("Step 5: Activate virtual environment", styles["H2"]))
    code("venv\\Scripts\\activate")

    story.append(Paragraph("Step 6: Install core packages", styles["H2"]))
    code("pip install -r requirements.txt")

    story.append(Paragraph("Step 7: Install local mode packages (optional)", styles["H2"]))
    code("pip install -r requirements-local.txt")
    note("Only needed if GENERATION_MODE=local in .env. Skippable for API mode.")

    story.append(Paragraph("Step 8: Setup .env", styles["H2"]))
    code("copy .env.example .env")
    body("Edit .env in Notepad and fill in your API keys.")

    story.append(Paragraph("Step 9: Start the application", styles["H2"]))
    code("python app.py")
    body('Open browser: <font face="Courier" color="#1b5e20">http://127.0.0.1:5000</font>')

    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="60%", thickness=1, color=DARK_BLUE))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "Forensic AI Investigation System - Setup Guide",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=9, textColor=MID_BLUE, alignment=TA_CENTER)
    ))

    # ── Build PDF ──
    doc.build(story)
    print(f"\n  Setup guide generated: {output}")
    print(f"  Size: {output.stat().st_size / 1024:.0f} KB")
    return output


if __name__ == "__main__":
    build_pdf()
