"""
index_kb.py - Build BFSI ChromaDB knowledge base
Place at: C:/AIManpres2/app/index_kb.py
Run once (with venv active): python index_kb.py  (from inside app/ folder)

Creates ChromaDB vector store in kb_store/
Uses sentence-transformers for local embeddings (no API key needed)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env with encoding fallback (Windows Notepad saves UTF-16 by default)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")

BASE_DIR  = Path(__file__).parent.parent
KB_DOCS   = BASE_DIR / "kb_docs"
KB_STORE  = BASE_DIR / "kb_store"
KB_STORE.mkdir(exist_ok=True)

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"    # ~90MB, fast, good quality
COLLECTION  = "apex_bank_bfsi"

# ── Seed knowledge base content ───────────────────────────────────────────────
# These are written to kb_docs\ if not already present.
# Add your own .txt files to kb_docs\ for richer answers.

SEED_DOCS = {
    "savings_accounts.txt": """
Apex Bank Savings Account Products

Regular Savings Account:
- Minimum balance: Rs. 1,000 (Metro), Rs. 500 (Semi-Urban/Rural)
- Interest rate: 3.5% per annum on daily balance
- Free debit card with Rs. 40,000 daily ATM limit
- Free NEFT/IMPS/RTGS via net banking
- Penalty for non-maintenance: Rs. 100 per quarter

Premium Savings Account (ApexPlus):
- Minimum balance: Rs. 25,000
- Interest rate: 4.0% per annum
- Relationship Manager assigned
- Free demand drafts (up to 10 per month)
- Complimentary airport lounge access (2 per quarter)
- Preferential forex rates

Salary Account:
- Zero minimum balance for salary accounts
- Overdraft facility up to 2x net monthly salary
- Free unlimited ATM transactions at any bank
- Interest rate: 3.5% per annum
""",

    "fixed_deposits.txt": """
Apex Bank Fixed Deposit (FD) Rates — Effective June 2025

Domestic FD Rates (Below Rs. 2 Crore):
- 7 days to 29 days: 3.00% p.a.
- 30 days to 90 days: 4.25% p.a.
- 91 days to 180 days: 5.00% p.a.
- 181 days to less than 1 year: 5.75% p.a.
- 1 year to less than 2 years: 6.80% p.a.
- 2 years to less than 3 years: 6.70% p.a.
- 3 years to less than 5 years: 6.60% p.a.
- 5 years and above: 6.50% p.a.

Senior Citizen Rates: Additional 0.50% p.a. on all tenures.
Tax Saver FD: 5-year lock-in, eligible for 80C deduction up to Rs. 1.5 lakh.

Premature withdrawal penalty: 1% on contracted rate.
Minimum FD amount: Rs. 10,000.
Auto-renewal: Available. Choose at time of booking.
""",

    "home_loans.txt": """
Apex Bank Home Loan Products

Standard Home Loan:
- Interest rate: Starting at 8.65% p.a. (floating, linked to RLLR)
- Maximum tenure: 30 years
- Maximum loan amount: Rs. 10 crore
- LTV ratio: Up to 90% for loans up to Rs. 30 lakh; 80% up to Rs. 75 lakh; 75% above Rs. 75 lakh
- Processing fee: 0.5% of loan amount (minimum Rs. 5,000)
- No prepayment penalty on floating rate loans

Pradhan Mantri Awas Yojana (PMAY) linked loans available.
Credit score requirement: Minimum 700 CIBIL score recommended.

Documents required:
- KYC documents (Aadhaar, PAN)
- Latest 3 months salary slips or 2 years ITR for self-employed
- Bank statement (6 months)
- Property documents

Personal Loan:
- Rate: 11.5% to 18% p.a. based on profile
- Amount: Rs. 50,000 to Rs. 25 lakh
- Tenure: 12 to 60 months
- Disbursal in 24 hours for pre-approved customers
""",

    "credit_cards.txt": """
Apex Bank Credit Cards

ApexRewards Classic:
- Annual fee: Rs. 499 (waived on Rs. 1 lakh annual spend)
- Reward rate: 1 point per Rs. 100 spend
- Point value: 0.25 paise per point
- Fuel surcharge waiver: 1% at all fuel stations
- Interest rate: 3.6% per month (43.2% p.a.)
- Minimum payment: 5% of outstanding or Rs. 200 whichever is higher

ApexPrime Signature:
- Annual fee: Rs. 2,999 (waived on Rs. 4 lakh annual spend)
- Reward rate: 3 points per Rs. 100 on dining/travel; 1.5 on others
- Airport lounge access: 8 free visits per year (domestic + international)
- Complimentary golf sessions: 2 per month
- Travel insurance cover: Rs. 50 lakh

ApexCashback Card:
- Annual fee: Rs. 999
- Cashback: 5% on online spends (up to Rs. 500/month), 1% on all others
- No rewards points — direct cashback to statement

Credit card limit: Based on income and credit score. Minimum Rs. 20,000.
""",

    "kyc_and_digital.txt": """
Apex Bank KYC and Digital Services

KYC Process:
- New customers: Aadhaar-based eKYC available online — takes 5 minutes
- Re-KYC: Required every 2 years for high-risk customers, 10 years for low-risk
- KYC status check: Available on ApexMobile app or by calling 1800-123-4567
- Pending KYC: Account will have restrictions on transactions above Rs. 50,000

Net Banking (ApexNet):
- Register at apexbank.in with account number and debit card
- NEFT/IMPS/RTGS available 24x7
- Fixed deposit booking, loan repayment, and credit card payment available online
- International transaction limit: Set via app

ApexMobile App:
- Available on iOS and Android
- UPI ID: accountnumber@apexbank
- Features: Statement download, cheque book request, card block/unblock, loan EMI, FD booking

Customer Care:
- Toll-free: 1800-123-4567 (24x7)
- Email: support@apexbank.in
- Home branch: Anywhere in India through ApexAnywhere programme
""",

    "insurance_and_investments.txt": """
Apex Bank Wealth and Insurance Products

Life Insurance (tied up with ApexLife):
- Term plan: Pure protection from Rs. 50 lakh to Rs. 5 crore cover
- Premium example: Rs. 5 crore cover for 30-year-old non-smoker male = approx Rs. 12,000/year
- ULIP plans available for market-linked growth

Health Insurance (ApexHealth Shield):
- Individual: Rs. 3 lakh to Rs. 1 crore cover
- Family floater available
- Cashless at 6,000+ network hospitals across India
- Pre and post-hospitalisation cover: 30 and 60 days

Mutual Funds (through Apex Securities):
- SIP starting Rs. 500/month
- Curated baskets: Conservative (Debt), Balanced, Growth (Equity)
- Demat account opening: Free for bank customers

National Pension Scheme (NPS):
- Apex Bank is a registered POP for NPS
- Open NPS account at any branch with Aadhaar
- Tax benefit: Up to Rs. 50,000 additional under 80CCD(1B)
"""
}


def write_seed_docs():
    """Write seed documents to kb_docs/ if not already present."""
    KB_DOCS.mkdir(exist_ok=True)
    written = 0
    for fname, content in SEED_DOCS.items():
        path = KB_DOCS / fname
        if not path.exists():
            path.write_text(content.strip(), encoding="utf-8")
            print(f"  Created seed doc: {fname}")
            written += 1
    if written == 0:
        print("  Seed docs already present — skipping.")
    return written


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) > 20]


def build_index():
    print("\n=== Apex Bank BFSI Knowledge Base Indexer ===\n")

    # 1. Write seed docs if needed
    print("Step 1: Checking seed documents…")
    write_seed_docs()

    # 2. Load embedding model
    print(f"\nStep 2: Loading embedding model ({EMBED_MODEL})…")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("  Model loaded.")

    # 3. Init ChromaDB
    print(f"\nStep 3: Initialising ChromaDB at {KB_STORE}…")
    client = chromadb.PersistentClient(path=str(KB_STORE))
    
    # Delete and recreate collection to ensure fresh index
    try:
        client.delete_collection(COLLECTION)
        print("  Deleted existing collection for fresh rebuild.")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)
    print(f"  Collection '{COLLECTION}' ready.")

    # 4. Index documents
    print(f"\nStep 4: Indexing documents from {KB_DOCS}…")
    doc_files = list(KB_DOCS.glob("*.txt")) + list(KB_DOCS.glob("*.md"))
    if not doc_files:
        print("  No documents found in kb_docs/ — exiting.")
        return

    all_chunks, all_ids, all_metas = [], [], []
    for doc_path in doc_files:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            cid = f"{doc_path.stem}_{i:04d}"
            all_chunks.append(chunk)
            all_ids.append(cid)
            all_metas.append({"source": doc_path.name, "chunk": i})
        print(f"  {doc_path.name}: {len(chunks)} chunks")

    # 5. Embed and upsert in batches
    print(f"\nStep 5: Generating embeddings for {len(all_chunks)} chunks…")
    BATCH = 64
    for start in range(0, len(all_chunks), BATCH):
        batch_chunks = all_chunks[start:start + BATCH]
        batch_ids    = all_ids[start:start + BATCH]
        batch_metas  = all_metas[start:start + BATCH]
        embeddings   = embedder.encode(batch_chunks, show_progress_bar=False).tolist()
        collection.add(documents=batch_chunks, embeddings=embeddings, ids=batch_ids, metadatas=batch_metas)
        print(f"  Indexed {min(start + BATCH, len(all_chunks))}/{len(all_chunks)}", end="\r")

    print(f"\n  Done. {len(all_chunks)} chunks indexed.")
    print(f"\n=== Index built successfully at {KB_STORE} ===")
    print("\nNext: Start the server with:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n")


if __name__ == "__main__":
    build_index()
