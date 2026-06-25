import os
import re
import json
import fitz
import cv2 as cv
import numpy as np
import pandas as pd
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq 
import base64

load_dotenv()

app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:5173",          
    "http://127.0.0.1:5173",         
    "https://textvlm.netlify.app"     
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "static_downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

groq_key = os.getenv("GROQ_API_KEY")
groq_client = None

if groq_key:
    try:
        groq_client = Groq(api_key=groq_key)
    except Exception as e:
        print(f"Groq Client Init Warning: {e}")
else:
    print("CRITICAL WARNING: GROQ_API_KEY is missing from your environment config.")

def preprocess_image(pix_bytes) -> str:
    """Performs normalization and returns a clean, base64-encoded string for Groq."""
    nparr = np.frombuffer(pix_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    
    if img is None or img.size == 0:
        return base64.b64encode(pix_bytes).decode("utf-8")

    # Contrast adjustment
    contrast_img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
    # Binarization, Deskew
    _, binary_img = cv.threshold(contrast_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    try:
        cv.bitwise_not(binary_img, binary_img)
        coords = np.column_stack(np.where(binary_img > 0))
        if len(coords) > 0:
            rect = cv.minAreaRect(coords)
            angle = rect[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            (h, w) = binary_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv.getRotationMatrix2D(center, angle, 1.0)
            cv.bitwise_not(binary_img, binary_img)
            deskewed_img = cv.warpAffine(binary_img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=255)
        else:
            cv.bitwise_not(binary_img, binary_img)
            deskewed_img = binary_img
    except Exception:
        if len(binary_img.shape) > 0:
            cv.bitwise_not(binary_img, binary_img)
        deskewed_img = binary_img

    # Compress to JPEG format to keep payload small and fast
    _, encoded_img = cv.imencode(".jpg", deskewed_img, [int(cv.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(encoded_img.tobytes()).decode("utf-8")

def extract_structured_data_via_groq(base64_image: str) -> list:
    if not groq_client:
        raise HTTPException(status_code=401, detail="Groq cloud engine initialization failed.")

    prompt = """You are an expert OCR analyst processing a scanned handwritten attendance sheet from a university in either Africa or India.

    DOCUMENT STRUCTURE:
    This is a multi-row attendance table with exactly these columns in order:
    1. Full Name (handwritten, left column — often multi-line for long names)
    2. Learner Type (checkbox column — look for a checkmark or tick next to "Student" or "Faculty")
    3. Email (handwritten — domain varies: .edu, .ac.in, .edu.gh, .ac.ke, gmail.com, outlook.com, yahoo.com, etc.)
    4. Phone (handwritten — format varies by country; may include country codes like +233, +91, +234, +254, +256, +27, etc.)
    5. Preferred Contact (checkbox — "Email" and/or "Phone" boxes; extract whichever are ticked)
    6. Signature (rightmost column — handwritten mark or initials)

    REGIONAL & SPELLING CONTEXT:
    Names will be from one of two broad regional pools — do not Westernize or anglicize either:

    West/East/Southern African patterns:
    - Ghanaian: Kofi, Kwame, Ama, Nii, Adwoa, Yaa, Ato, Afia, Okyere, Nana, Abena, Akosua; surnames like Asante, Boahen, Tettey, Adu, Asare, Dadzie, Mensah, Osei, Aziablame.
    - Nigerian: prefixes Chukwu-, Olu-, Ade-, Nkem-; suffixes -yemi, -tunde, -bola, -chi.
    - Francophone African: hyphenated or compound surnames like Tanoh-Rivers, Maikano-Lawal, Nkoa-Bessala; names ending in -ou, -ié.
    - East African (Kenyan, Ugandan, Tanzanian): Amina, Fatuma, Wanjiru, Kipchoge, Odhiambo, Mutua, Achieng.

    South Asian (Indian) patterns:
    - Common first names: Aarav, Rohan, Priya, Ananya, Arjun, Sneha, Rahul, Kavya, Vikram, Divya, Ishaan, Pooja.
    - South Indian surnames: Krishnamurthy, Venkataraman, Subramaniam, Iyer, Pillai, Nair, Reddy, Naidu.
    - North Indian surnames: Sharma, Verma, Singh, Gupta, Mishra, Yadav, Tiwari, Joshi, Patel, Shah.
    - Names may be written in varied orders (given name first or family name first).

    TRANSCRIPTION RULES:
    1. Row-by-row: process each table row independently. One row = one participant.
    2. Multi-line names: if a name wraps to a second line within the name cell, join them as a single full_name string.
    3. Learner type: the column has two printed options — "Student" and "Faculty". A checkbox or tick marks the correct one. If the tick is ambiguous or missing, default to "Student".
    4. Email cleanup:
    - Fix common handwriting/OCR confusion: '0' vs 'o', '1' vs 'l' vs 'i', '6' vs 'b', rn vs m, '@' misread as 'a' or 'o'.
    - Normalize malformed domains to their most plausible form (e.g. 'gmial.com' → 'gmail.com', 'outlok.com' → 'outlook.com').
    - Do not assume or invent a domain — only correct what is clearly a transcription artifact.
    - If two emails are written (e.g. personal + institutional), capture both separated by ' / '.
    - If no email is written, return "".
    5. Phone cleanup:
    - Preserve the full number including country code if written (e.g. +91, +233, +234, +254).
    - Remove internal spaces or dashes within the number but keep the leading +.
    - If no phone is written, return "".
    6. Preferred contact: return exactly one of — "Email", "Phone", or "Email / Phone" — based on which checkboxes are ticked.
    7. Signature: return true if ANY handwritten mark, squiggle, initials, or stroke is present inside the signature box. Return false only if the box is completely blank.
    8. Skip any row where the Full Name cell is empty or contains only a printed placeholder like "JOHN DOE".
    9. Do NOT hallucinate or invent data for any field. If a field is genuinely blank, return "".

    OUTPUT — return ONLY a raw JSON array, no markdown, no backticks, no preamble, no explanation:
    {
        "full_name": "Exact transcribed name",
        "learner_type": "Student" or "Faculty",
        "email": "cleaned email or empty string",
        "phone": "full phone number or empty string",
        "preferred_contact": "Email" | "Phone" | "Email / Phone",
        "has_signature": true or false
    }
    """

    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,  
            max_tokens=2048
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # markdown cleanup
        clean_json = re.sub(r"```json|```", "", response_text).strip()
        return json.loads(clean_json)

    except json.JSONDecodeError as je:
        print(f"JSON Structure error from model output: {response_text}")
        raise HTTPException(status_code=502, detail="Groq output could not be parsed into clean JSON.")
    except Exception as e:
        print(f"Groq Cloud processing drop error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Groq runtime issue: {str(e)}")

@app.get("/")
async def root_health_check():
    return {"status": "online", "application": "PDF VLM Engine"}


@app.post("/process-attendance")
async def process_attendance(files: List[UploadFile] = File(...)):
    if not groq_client:
        raise HTTPException(status_code=401, detail="Missing valid GROQ_API_KEY on local server.")

    all_records = []
    total_pages_count = 0

    for file in files:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(1, len(doc)):
            total_pages_count += 1
            print(f"Processing page {page_num} of {len(doc)-1} from file: {file.filename}")
            
            page = doc.load_page(page_num)
            pix = page.get_pixmap() 
            
            # Preprocess to base64 string
            base64_img_str = preprocess_image(pix.tobytes())
            
            page_data = extract_structured_data_via_groq(base64_img_str)
            
            for item in page_data:
                if item.get("full_name") and "JOHN DOE" not in item.get("full_name").upper():
                    all_records.append(item)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=["full_name", "learner_type", "email", "phone", "preferred_contact", "has_signature"])
    
    total_people = len(df)
    total_signed = int(df["has_signature"].sum()) if total_people > 0 else 0
    total_unsigned = total_people - total_signed

    csv_filename = "attendance_report.csv"
    excel_filename = "attendance_report.xlsx"
    
    df.to_csv(os.path.join(OUTPUT_DIR, csv_filename), index=False)
    df.to_excel(os.path.join(OUTPUT_DIR, excel_filename), index=False, engine='openpyxl')

    return {
        "summary": {
            "total_files": len(files),
            "total_pages": total_pages_count,
            "total_people": total_people,
            "total_signed": total_signed,
            "total_unsigned": total_unsigned
        },
        "downloads": {
            "csv": f"/static/{csv_filename}",
            "excel": f"/static/{excel_filename}"
        }
    }
