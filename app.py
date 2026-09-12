"""
JabFraud — Check the job before you apply.

A single-file Streamlit app that takes a job-posting URL and returns an
explainable 0-100 fraud-risk score, combining:
  1. Direct HTML scraping (requests + BeautifulSoup + trafilatura + extruct/JSON-LD)
  2. A browser fallback (Playwright) with screenshot + OCR (pytesseract) for
     JavaScript-heavy pages that the direct method can't read
  3. A rule-based fraud checklist (company / contact / description / application signals)
  4. An AI review using the Grok API (xAI, OpenAI-compatible endpoint)

Deployment note: the Playwright + OCR fallback needs OS-level libraries
(a Chromium runtime and the `tesseract-ocr` binary) that pip alone cannot
install. On Streamlit Community Cloud this normally means adding a
`packages.txt` file alongside this app. Without it, the app still works —
it just won't be able to fall back to the headless-browser / OCR path for
JavaScript-only pages, and will say so instead of crashing.
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

import trafilatura

try:
    import extruct
    from w3lib.html import get_base_url
    EXTRUCT_AVAILABLE = True
except Exception:
    EXTRUCT_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except Exception:
    OPENAI_SDK_AVAILABLE = False


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

APP_TITLE = "JabFraud"
APP_TAGLINE = "Check the job before you apply."

# xAI's Grok endpoint is OpenAI-compatible.
GROK_BASE_URL = "https://api.x.ai/v1"
# Model names change fairly often on xAI's side — override via
# st.secrets["GROK_MODEL"] if this stops being valid.
DEFAULT_GROK_MODEL = "grok-4-fast-reasoning"

REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 JabFraud/1.0"
)

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com", "yandex.com", "gmx.com",
}

URGENCY_PHRASES = [
    "hire immediately", "urgent hiring", "urgent vacancy", "no experience required",
    "no experience needed", "limited slots", "immediate start",
    "immediate joining", "act now", "start today", "same day hiring",
]
# Note: "Apply Now" is deliberately excluded — it's a standard button label on
# almost every legitimate job board and would otherwise cause constant false positives.

# Common two-part ccTLD suffixes where the "registrable domain" needs three
# labels instead of two (e.g. acme.co.uk, not co.uk). Not exhaustive, but
# covers the common cases well enough for a domain-similarity heuristic.
_MULTI_PART_TLDS = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "com.pk", "com.au",
    "co.jp", "com.br", "co.nz", "co.za", "com.sg", "com.hk",
}


def _root_domain(netloc: str) -> str:
    """Naive registrable-domain extraction, e.g. apply.acmecorp.com -> acmecorp.com."""
    netloc = (netloc or "").lower().split(":")[0]
    parts = netloc.split(".")
    if len(parts) <= 2:
        return netloc
    last_two = ".".join(parts[-2:])
    if last_two in _MULTI_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two

PAYMENT_RED_FLAG_PHRASES = [
    "registration fee", "processing fee", "training fee", "security deposit",
    "bank details", "bank account number", "social security number", "ssn",
    "pay before you start", "send money", "western union", "wire transfer",
    "cryptocurrency payment", "gift card",
]

FREE_WEBSITE_HOSTS = [
    "wixsite.com", "blogspot.com", "weebly.com", "wordpress.com",
    "sites.google.com", "carrd.co", "godaddysites.com",
]


# --------------------------------------------------------------------------
# Playwright / OCR readiness (best-effort, never fatal)
# --------------------------------------------------------------------------

def ensure_chromium_installed() -> bool:
    """Try to make sure Playwright's Chromium is available. Returns True if usable."""
    marker = "/tmp/.jabfraud_chromium_ready"
    if os.path.exists(marker):
        return True
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=240,
        )
        if result.returncode == 0:
            with open(marker, "w") as f:
                f.write("ok")
            return True
        return False
    except Exception:
        return False


# --------------------------------------------------------------------------
# Stage 1 — Reading the job page
# --------------------------------------------------------------------------

def fetch_direct(url: str) -> dict:
    """The easy way: plain HTTP GET + HTML parsing. No JavaScript execution."""
    out = {
        "method": "direct",
        "success": False,
        "html": "",
        "text": "",
        "structured": {},
        "final_url": url,
        "status_code": None,
        "redirect_chain": [],
        "error": None,
    }
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        out["status_code"] = resp.status_code
        out["final_url"] = resp.url
        out["redirect_chain"] = [r.url for r in resp.history]
        if resp.status_code >= 400:
            out["error"] = f"HTTP {resp.status_code}"
            return out

        html = resp.text
        out["html"] = html

        # Main readable text, ads/menus stripped out.
        extracted_text = trafilatura.extract(html, url=resp.url) or ""
        out["text"] = extracted_text.strip()

        # Hidden structured data (Schema.org / JSON-LD) — often more reliable
        # than the visible text, and used by systems like Google Jobs.
        if EXTRUCT_AVAILABLE:
            try:
                base_url = get_base_url(html, resp.url)
                data = extruct.extract(html, base_url=base_url, syntaxes=["json-ld", "microdata"])
                out["structured"] = data
            except Exception:
                out["structured"] = {}

        # A page counts as "successfully read" if there's a meaningful amount
        # of text, or structured JobPosting data was found.
        has_structured_job = _find_jobposting(out["structured"]) is not None
        out["success"] = len(out["text"]) > 200 or has_structured_job
        return out
    except requests.RequestException as e:
        out["error"] = str(e)
        return out


def fetch_with_browser(url: str) -> dict:
    """The backup way: a real (headless) browser, for JS-heavy pages."""
    out = {
        "method": "browser",
        "success": False,
        "html": "",
        "text": "",
        "structured": {},
        "final_url": url,
        "screenshot": None,
        "ocr_text": "",
        "error": None,
    }
    if not ensure_chromium_installed():
        out["error"] = (
            "Headless browser isn't available in this environment (Chromium/OS "
            "libraries missing). Add a packages.txt with Playwright's system "
            "dependencies to enable this fallback."
        )
        return out

    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        out["error"] = f"Playwright not importable: {e}"
        return out

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass

            # Scroll to trigger lazy-loaded content.
            for _ in range(4):
                page.mouse.wheel(0, 1500)
                page.wait_for_timeout(400)

            out["final_url"] = page.url
            html = page.content()
            out["html"] = html

            screenshot_bytes = page.screenshot(full_page=True)
            out["screenshot"] = screenshot_bytes

            browser.close()

        extracted_text = trafilatura.extract(html, url=out["final_url"]) or ""
        out["text"] = extracted_text.strip()

        if EXTRUCT_AVAILABLE:
            try:
                base_url = get_base_url(html, out["final_url"])
                out["structured"] = extruct.extract(html, base_url=base_url, syntaxes=["json-ld", "microdata"])
            except Exception:
                out["structured"] = {}

        has_structured_job = _find_jobposting(out["structured"]) is not None
        out["success"] = len(out["text"]) > 200 or has_structured_job

        # If text is still thin, fall back to OCR on the screenshot.
        if not out["success"] and OCR_AVAILABLE and out["screenshot"]:
            try:
                img = Image.open(io.BytesIO(out["screenshot"]))
                ocr_text = pytesseract.image_to_string(img)
                out["ocr_text"] = ocr_text.strip()
                if len(out["ocr_text"]) > 200:
                    out["success"] = True
                    if not out["text"]:
                        out["text"] = out["ocr_text"]
            except Exception as e:
                out["error"] = f"OCR failed: {e}"

        return out
    except Exception as e:
        out["error"] = str(e)
        return out


# --------------------------------------------------------------------------
# Stage 2 — Turning raw page data into structured job fields
# --------------------------------------------------------------------------

def _find_jobposting(structured: dict):
    """Look through extruct output for a Schema.org JobPosting block."""
    if not structured:
        return None
    for block in structured.get("json-ld", []) or []:
        t = block.get("@type")
        if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
            return block
        # Some sites wrap it in @graph
        for node in block.get("@graph", []) or []:
            nt = node.get("@type")
            if nt == "JobPosting" or (isinstance(nt, list) and "JobPosting" in nt):
                return node
    for block in structured.get("microdata", []) or []:
        if block.get("type", "").endswith("JobPosting"):
            return block.get("properties", {})
    return None


def _text_of(value):
    """Schema.org fields are sometimes plain strings, sometimes dicts/lists."""
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"<[^>]+>", " ", value).strip()
    if isinstance(value, dict):
        for key in ("name", "value", "@value", "streetAddress", "addressLocality"):
            if key in value:
                return _text_of(value[key])
        return " ".join(_text_of(v) for v in value.values() if isinstance(v, (str, dict)))
    if isinstance(value, list):
        return ", ".join(_text_of(v) for v in value)
    return str(value)


EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{3,4}")
SALARY_RE = re.compile(
    r"(?:\$|USD|PKR|Rs\.?|₹|£|€)\s?[\d,]+(?:\.\d+)?\s?(?:-|to)?\s?(?:\$|USD|PKR|Rs\.?|₹|£|€)?\s?[\d,]*\s*"
    r"(?:/\s?(?:year|yr|month|mo|hour|hr))?",
    re.IGNORECASE,
)


def parse_job_fields(fetch_result: dict, original_url: str) -> dict:
    """Combine structured data + raw text + HTML into the job-detail fields
    described in the PRD (title, company, description, location, salary,
    employment type, experience/skills, benefits, recruiter contact,
    application URL, company URL, domain, redirects, evidence)."""

    html = fetch_result.get("html", "")
    text = fetch_result.get("text", "") or fetch_result.get("ocr_text", "")
    structured = fetch_result.get("structured", {})
    final_url = fetch_result.get("final_url", original_url)

    soup = BeautifulSoup(html, "lxml") if html else None
    job = _find_jobposting(structured) or {}

    def sd(*keys, default=""):
        node = job
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return _text_of(node) or default

    # --- core fields, structured data first, then HTML/text heuristics ---
    title = sd("title") or (soup.title.text.strip() if soup and soup.title else "")
    company = sd("hiringOrganization", "name")
    location = sd("jobLocation", "address") or sd("jobLocation")
    salary = sd("baseSalary", "value", "value") or sd("baseSalary")
    employment_type = sd("employmentType")
    description_html = job.get("description", "") if isinstance(job, dict) else ""
    description = _text_of(description_html) if description_html else ""

    if not description:
        description = text[:6000]

    if not salary:
        m = SALARY_RE.search(text)
        salary = m.group(0).strip() if m else ""

    if not location and soup:
        loc_meta = soup.find("meta", attrs={"name": "job_location"})
        location = loc_meta["content"] if loc_meta and loc_meta.get("content") else ""

    if not company and soup:
        og_site = soup.find("meta", property="og:site_name")
        company = og_site["content"].strip() if og_site and og_site.get("content") else ""
    if not company:
        company = urlparse(final_url).netloc.replace("www.", "")

    # --- detect "this is a job board's listing page, not a single job" ---
    job_link_count = 0
    if soup:
        job_link_pattern = re.compile(r"/jobs?/[\w-]+", re.IGNORECASE)
        job_hrefs = {a["href"] for a in soup.find_all("a", href=True) if job_link_pattern.search(a["href"])}
        job_link_count = len(job_hrefs)
    looks_like_listing_page = (not job.get("title")) and job_link_count >= 6

    # --- contact info ---
    emails = sorted(set(EMAIL_RE.findall(text)))
    recruiter_email = emails[0] if emails else ""
    phones = sorted(set(m.strip() for m in PHONE_RE.findall(text) if len(re.sub(r"\D", "", m)) >= 7))
    recruiter_phone = phones[0] if phones else ""

    # --- links ---
    application_url = ""
    company_url = sd("hiringOrganization", "sameAs")
    external_links = []
    if soup:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            label = (a.get_text() or "").strip().lower()
            full = urljoin(final_url, href)
            if not application_url and ("apply" in label or "apply" in href.lower()):
                application_url = full
            netloc = urlparse(full).netloc
            page_netloc = urlparse(final_url).netloc
            if netloc and netloc != page_netloc and "http" in full:
                external_links.append(full)
    external_links = sorted(set(external_links))[:15]

    domain = urlparse(final_url).netloc

    # --- experience / skills / benefits (simple keyword heuristics over description) ---
    exp_match = re.search(r"(\d+)\+?\s*(?:-\s*\d+\s*)?years?\s+(?:of\s+)?experience", description, re.IGNORECASE)
    experience = exp_match.group(0) if exp_match else ""

    skill_keywords = [
        "python", "java", "javascript", "sql", "excel", "communication",
        "react", "node", "aws", "marketing", "sales", "customer service",
        "project management", "accounting", "design", "figma",
    ]
    skills_found = sorted({kw for kw in skill_keywords if kw in description.lower()})

    benefit_keywords = [
        "health insurance", "paid leave", "remote work", "work from home",
        "bonus", "401k", "pension", "flexible hours", "medical",
        "paid time off", "life insurance",
    ]
    benefits_found = sorted({kw for kw in benefit_keywords if kw in description.lower()})

    return {
        "title": title.strip(),
        "company": company.strip(),
        "description": description.strip(),
        "location": location.strip(),
        "salary": salary.strip(),
        "employment_type": employment_type.strip(),
        "experience": experience,
        "skills": skills_found,
        "benefits": benefits_found,
        "recruiter_email": recruiter_email,
        "recruiter_phone": recruiter_phone,
        "application_url": application_url,
        "company_url": company_url,
        "page_domain": domain,
        "final_url": final_url,
        "redirect_chain": fetch_result.get("redirect_chain", []),
        "external_links": external_links,
        "raw_text": text,
        "screenshot": fetch_result.get("screenshot"),
        "looks_like_listing_page": looks_like_listing_page,
    }


def compute_extraction_confidence(method: str, ocr_used: bool, job: dict) -> int:
    """A rough, honest confidence score — not a precise statistic."""
    if ocr_used:
        base = 81
    elif method == "browser":
        base = 87
    else:
        base = 92

    important = [job["title"], job["company"], job["description"], job["location"]]
    missing = sum(1 for f in important if not f)
    base -= missing * 8
    return max(35, min(97, base))


# --------------------------------------------------------------------------
# Stage 3a — Rule-based fraud checklist
# --------------------------------------------------------------------------

def run_rule_checklist(job: dict) -> tuple:
    """Returns (rule_score 0-100, list of flag dicts {icon, title, detail})."""
    flags = []
    score = 0
    text_lower = (job.get("description", "") + " " + job.get("raw_text", "")).lower()

    # --- Company signals ---
    if not job.get("company") or job["company"].lower() in job.get("page_domain", "").lower() and len(job["company"]) < 4:
        flags.append({"icon": "⚠", "title": "Missing or unclear company name",
                       "detail": "No clearly identifiable hiring company was found on the page."})
        score += 20

    if job.get("company_url"):
        host = urlparse(job["company_url"]).netloc.lower()
        if any(free in host for free in FREE_WEBSITE_HOSTS):
            flags.append({"icon": "⚠", "title": "Company site on a free website builder",
                           "detail": f"The listed company site ({host}) is hosted on a free builder rather than its own domain."})
            score += 15

    # --- Contact signals ---
    email = job.get("recruiter_email", "")
    if email:
        email_domain = email.split("@")[-1].lower()
        if email_domain in FREE_EMAIL_DOMAINS:
            flags.append({"icon": "⚠", "title": "Personal email address",
                           "detail": f"The recruiter used a personal address ({email_domain}) instead of a company email."})
            score += 20
        else:
            page_root = _root_domain(job.get("page_domain", ""))
            email_root = _root_domain(email_domain)
            if page_root and email_root and page_root != email_root:
                flags.append({"icon": "⚠", "title": "Recruiter email doesn't match company domain",
                               "detail": f"Contact email domain ({email_domain}) differs from the job page's domain ({job.get('page_domain', '')})."})
                score += 15

    # --- Job description signals ---
    if len(job.get("description", "")) < 150:
        flags.append({"icon": "⚠", "title": "Vague or very short description",
                       "detail": "The job description is unusually short on real detail about the role."})
        score += 12

    urgent_hits = [p for p in URGENCY_PHRASES if p in text_lower]
    if urgent_hits:
        flags.append({"icon": "⚠", "title": "Urgency language",
                       "detail": f"The post pressures the reader to act immediately (e.g. \"{urgent_hits[0]}\")."})
        score += 12

    payment_hits = [p for p in PAYMENT_RED_FLAG_PHRASES if p in text_lower]
    if payment_hits:
        flags.append({"icon": "⚠", "title": "Requests payment or sensitive financial info",
                       "detail": f"Language resembling a payment or sensitive-data request was found (\"{payment_hits[0]}\")."})
        score += 30

    if "no experience required" in text_lower or "no experience needed" in text_lower:
        salary_digits = re.sub(r"[^\d]", "", job.get("salary", ""))
        if salary_digits and int(salary_digits[:6] or 0) > 50000:
            flags.append({"icon": "⚠", "title": "Unrealistic salary for stated experience",
                           "detail": "A high salary is offered despite explicitly requiring no experience."})
            score += 18

    # --- Application signals ---
    if job.get("application_url"):
        apply_host = urlparse(job["application_url"]).netloc.lower()
        page_host = (job.get("page_domain") or "").lower()
        company_host = urlparse(job.get("company_url") or "").netloc.lower()
        apply_root = _root_domain(apply_host)
        page_root = _root_domain(page_host)
        company_root = _root_domain(company_host) if company_host else ""
        # Also allow well-known third-party ATS platforms (Greenhouse, Lever,
        # Workday, etc.) without flagging them as "external" — those are how
        # most real companies host their application forms.
        known_ats_hosts = (
            "greenhouse.io", "lever.co", "myworkdayjobs.com", "icims.com",
            "smartrecruiters.com", "bamboohr.com", "breezy.hr", "ashbyhq.com",
            "jobvite.com", "taleo.net", "workable.com",
        )
        is_known_ats = any(apply_root.endswith(h) for h in known_ats_hosts)
        if apply_root and page_root and apply_root != page_root and apply_root != company_root and not is_known_ats:
            flags.append({"icon": "⚠", "title": "External application link",
                           "detail": f"The Apply link leads to {apply_host}, unrelated to the job page's own domain ({page_host})."})
            score += 15

    return min(100, score), flags


# --------------------------------------------------------------------------
# Stage 3b — AI analysis (Grok / xAI)
# --------------------------------------------------------------------------

def get_grok_client():
    api_key = st.session_state.get("grok_api_key") or st.secrets.get("GROK_API_KEY", "")
    if not api_key or not OPENAI_SDK_AVAILABLE:
        return None
    return OpenAI(api_key=api_key, base_url=GROK_BASE_URL)


AI_SYSTEM_PROMPT = (
    "You are a careful fraud-review assistant for job postings. Given extracted "
    "job details, decide how likely the posting is to be a scam or fraudulent. "
    "Respond ONLY with a single JSON object, no prose, no markdown fences, in "
    "exactly this shape:\n"
    '{"risk_score": <integer 0-100>, '
    '"findings": [{"issue": "<short title>", "explanation": "<one plain-English sentence>"}], '
    '"recommendation": "<one short plain-English sentence of advice for the job seeker>"}'
)


def run_ai_analysis(job: dict) -> dict:
    """Calls the Grok API and returns {risk_score, findings, recommendation, error}."""
    client = get_grok_client()
    if client is None:
        return {
            "risk_score": None, "findings": [], "recommendation": "",
            "error": "No Grok API key configured. Add GROK_API_KEY in Streamlit secrets "
                     "or enter it in the sidebar to enable AI analysis.",
        }

    model = st.secrets.get("GROK_MODEL", DEFAULT_GROK_MODEL)
    payload = {
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "salary": job.get("salary"),
        "employment_type": job.get("employment_type"),
        "description": (job.get("description") or "")[:4000],
        "recruiter_email": job.get("recruiter_email"),
        "recruiter_phone": job.get("recruiter_phone"),
        "application_url": job.get("application_url"),
        "company_url": job.get("company_url"),
        "page_domain": job.get("page_domain"),
        "external_links": job.get("external_links"),
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=30,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        risk_score = int(parsed.get("risk_score", 0))
        risk_score = max(0, min(100, risk_score))
        findings = parsed.get("findings", []) or []
        recommendation = parsed.get("recommendation", "")
        return {"risk_score": risk_score, "findings": findings, "recommendation": recommendation, "error": None}
    except json.JSONDecodeError:
        return {"risk_score": None, "findings": [], "recommendation": "",
                "error": "Grok responded, but not with valid JSON. The model may not support "
                         "structured JSON output — try a different GROK_MODEL."}
    except Exception as e:
        try:
            import openai as _openai_mod
            if isinstance(e, _openai_mod.AuthenticationError):
                msg = "Authentication failed — check that the Grok API key is correct."
            elif isinstance(e, _openai_mod.NotFoundError):
                msg = f"Model '{model}' was not found by the API — check GROK_MODEL is a valid, current xAI model name."
            elif isinstance(e, _openai_mod.RateLimitError):
                msg = "Rate limited by xAI — you're sending requests too fast or are out of quota."
            elif isinstance(e, _openai_mod.APIConnectionError):
                msg = ("Couldn't reach api.x.ai at all (network/firewall/VPN/proxy issue on this "
                       "machine, or no outbound internet access) — this happens before the API "
                       "key is even checked.")
            elif isinstance(e, _openai_mod.APIStatusError):
                msg = f"xAI API returned an error (HTTP {e.status_code}): {getattr(e, 'message', str(e))}"
            else:
                msg = str(e)
        except Exception:
            msg = str(e)
        return {"risk_score": None, "findings": [], "recommendation": "", "error": msg}


# --------------------------------------------------------------------------
# Stage 4 — Combined risk score
# --------------------------------------------------------------------------

def combine_scores(rule_score: int, ai_score) -> int:
    if ai_score is None:
        return rule_score
    return round(0.5 * rule_score + 0.5 * ai_score)


def risk_label(score: int) -> tuple:
    if score <= 30:
        return "🟢", "Low Risk", "#2ecc71"
    elif score <= 60:
        return "🟡", "Moderate Risk", "#f1c40f"
    elif score <= 80:
        return "🟠", "Suspicious", "#e67e22"
    else:
        return "🔴", "High Risk", "#e74c3c"


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="JabFraud — Check the job before you apply.", page_icon="🕵️", layout="centered")

with st.sidebar:
    st.markdown("### Settings")
    secret_key_present = bool(st.secrets.get("GROK_API_KEY", ""))
    if secret_key_present:
        st.success("Grok API key loaded from Streamlit secrets.")
    else:
        st.session_state["grok_api_key"] = st.text_input(
            "Grok API key (xAI)", type="password",
            help="Not stored anywhere except this browser session. "
                 "For a permanent setup, add GROK_API_KEY to Streamlit secrets instead.",
        )
    st.caption(
        "Playwright/OCR fallback needs OS packages (Chromium libs, tesseract-ocr) "
        "that aren't installable via requirements.txt alone. If those aren't present "
        "on this host, JabFraud still works for pages readable without a browser."
    )

st.title("🕵️ JabFraud")
st.caption(f"**{APP_TAGLINE}**")

url = st.text_input("Paste Job URL", placeholder="https://example.com/careers/job/12345")
analyze_clicked = st.button("🔍 Analyze Job", type="primary", use_container_width=True)

if analyze_clicked:
    if not url or not url.strip().lower().startswith(("http://", "https://")):
        st.error("Please paste a full job posting URL, starting with http:// or https://")
        st.stop()

    progress = st.status("Extracting job data…", expanded=True)

    fetch_result = fetch_direct(url)
    ocr_used = False

    if not fetch_result["success"]:
        progress.write("Dynamic content detected… switching to browser extraction.")
        browser_result = fetch_with_browser(url)
        if browser_result["success"] or browser_result.get("html"):
            fetch_result = browser_result
            ocr_used = bool(browser_result.get("ocr_text")) and not (browser_result.get("text") or "").strip()
        elif browser_result.get("error"):
            progress.write(f"Browser fallback unavailable: {browser_result['error']}")

    if not fetch_result.get("success") and not fetch_result.get("html"):
        progress.update(label="Could not read this page.", state="error")
        st.error(
            "JabFraud couldn't extract enough content from this URL with any available "
            "method. The site may be blocking automated access, or a headless-browser "
            "fallback isn't installed on this host."
        )
        if fetch_result.get("error"):
            st.caption(f"Details: {fetch_result['error']}")
        st.stop()

    progress.write("Pulling out key job details…")
    job = parse_job_fields(fetch_result, url)
    confidence = compute_extraction_confidence(fetch_result["method"], ocr_used, job)
    if job.get("looks_like_listing_page"):
        confidence = min(confidence, 40)

    if job.get("looks_like_listing_page"):
        st.warning(
            "This looks like a job-board **listing page** (lots of job links, no single "
            "job title found) rather than one specific posting. Results below will be "
            "unreliable — open one specific job from this list and paste that URL instead."
        )

    progress.write("Checking against the fraud checklist…")
    rule_score, rule_flags = run_rule_checklist(job)

    progress.write("Running AI fraud analysis…")
    ai_result = run_ai_analysis(job)

    final_score = combine_scores(rule_score, ai_result.get("risk_score"))
    icon, label, color = risk_label(final_score)

    progress.update(label="Fraud analysis completed", state="complete", expanded=False)

    # ---- Results ----
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:56px'>{icon}</div>"
            f"<div style='font-size:40px;font-weight:700;color:{color}'>{final_score}</div>"
            f"<div style='font-size:18px;font-weight:600;color:{color}'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Rule-based score", f"{rule_score} / 100")
        st.metric("AI score", f"{ai_result['risk_score']} / 100" if ai_result.get("risk_score") is not None else "—")
        st.caption(f"Extraction confidence: {confidence}% (method: {fetch_result['method']}{' + OCR' if ocr_used else ''})")

    if ai_result.get("error"):
        st.warning("AI analysis unavailable:")
        st.code(ai_result["error"], language=None)

    st.subheader("Why this score")
    all_findings = list(rule_flags)
    for f in ai_result.get("findings", []):
        all_findings.append({"icon": "⚠", "title": f.get("issue", "Flag"), "detail": f.get("explanation", "")})

    if not all_findings:
        st.success("No obvious warning signs were detected by either check.")
    else:
        for f in all_findings:
            st.markdown(f"**{f['icon']} {f['title']}** — {f['detail']}")

    recommendation = ai_result.get("recommendation") or (
        "Verify this position through the company's official careers page before applying."
    )
    st.info(f"**Recommendation:** {recommendation}")

    st.subheader("Job Details")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**Title:** {job['title'] or '—'}")
        st.markdown(f"**Company:** {job['company'] or '—'}")
        st.markdown(f"**Location:** {job['location'] or '—'}")
        st.markdown(f"**Salary:** {job['salary'] or '—'}")
        st.markdown(f"**Employment type:** {job['employment_type'] or '—'}")
        st.markdown(f"**Experience:** {job['experience'] or '—'}")
    with d2:
        st.markdown(f"**Recruiter email:** {job['recruiter_email'] or '—'}")
        st.markdown(f"**Recruiter phone:** {job['recruiter_phone'] or '—'}")
        st.markdown(f"**Application URL:** {job['application_url'] or '—'}")
        st.markdown(f"**Company URL:** {job['company_url'] or '—'}")
        st.markdown(f"**Page domain:** {job['page_domain'] or '—'}")
        if job["redirect_chain"]:
            st.markdown(f"**Redirects:** {' → '.join(job['redirect_chain'])}")

    if job["skills"]:
        st.markdown(f"**Skills mentioned:** {', '.join(job['skills'])}")
    if job["benefits"]:
        st.markdown(f"**Benefits mentioned:** {', '.join(job['benefits'])}")
    if job["external_links"]:
        with st.expander(f"External links found on the page ({len(job['external_links'])})"):
            for link in job["external_links"]:
                st.write(link)

    if job.get("screenshot"):
        st.subheader("Page Evidence")
        st.image(job["screenshot"], caption="Captured screenshot of the job page", use_container_width=True)

    with st.expander("Raw extracted text"):
        st.text(job["raw_text"][:5000] or "No text extracted.")

else:
    st.caption("Paste a job posting link above and click **Analyze Job** to get a risk report.")
