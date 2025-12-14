from markdown_pdf import Section, MarkdownPdf
import os

ARTIFACT_DIR = "/Users/nadeemyousaf/.gemini/antigravity/brain/6e4ad476-2e8e-476e-8003-faa6dbc5e719"
INPUT_MD = os.path.join(ARTIFACT_DIR, "comparison_study.md")
OUTPUT_PDF = os.path.join(ARTIFACT_DIR, "comparison_study.pdf")

TARGET_DIRS = [
    "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-ToN-IoT",
    "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-RT-IoT",
    "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-N-BaIoT"
]

def generate():
    print(f"Reading {INPUT_MD}...")
    with open(INPUT_MD, 'r') as f:
        content = f.read()
        
    print("Generating PDF...")
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(content, toc=False))
    pdf.save(OUTPUT_PDF)
    print(f"Saved {OUTPUT_PDF}")
    
    # Distribute
    for d in TARGET_DIRS:
        if os.path.exists(d):
            dest = os.path.join(d, "comparison_study.pdf")
            print(f"Copying to {dest}...")
            # Use os.system for simple copy or shutil, but pdf.save might lock? 
            # Re-saving logic or shutil copy
            import shutil
            shutil.copy(OUTPUT_PDF, dest)
        else:
            print(f"Warning: Output dir {d} does not exist.")

if __name__ == "__main__":
    generate()
