"""
JabFraud — Check the job before you apply.

A single-file Streamlit app that takes a job-posting URL and returns an
explainable 0-100 fraud-risk score, combining:

  1. Direct HTML scraping
     - requests
     - BeautifulSoup
     - trafilatura
     - extruct / JSON-LD / Microdata

  2. Browser fallback
     - Playwright
     - screenshot
     - OCR with pytesseract

  3. Rule-based fraud checklist

  4. AI review using Groq API

Deployment note:
The Playwright + OCR fallback needs OS-level packages.
On Streamlit Community Cloud, add a packages.txt file alongside app.py.

Without those packages, direct extraction still works.
"""

import io
import json
import os
import re
import random
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

import trafilatura


# --------------------------------------------------------------------------
# Optional imports
# --------------------------------------------------------------------------

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

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

REQUEST_TIMEOUT = 15

# A small pool of realistic desktop User-Agent strings. Rotating between
# these (instead of always sending the same one) reduces the odds of
# tripping the simplest User-Agent-based bot blocks.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 JabFraud/1.0",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.4 Safari/605.1.15",

    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
]


def _random_user_agent() -> str:
    return random.choice(USER_AGENTS)


FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "aol.com",
    "icloud.com",
    "mail.com",
    "protonmail.com",
    "yandex.com",
    "gmx.com",
}


URGENCY_PHRASES = [
    "hire immediately",
    "urgent hiring",
    "urgent vacancy",
    "no experience required",
    "no experience needed",
    "limited slots",
    "immediate start",
    "immediate joining",
    "act now",
    "start today",
    "same day hiring",
]


PAYMENT_RED_FLAG_PHRASES = [
    "registration fee",
    "processing fee",
    "training fee",
    "security deposit",
    "bank details",
    "bank account number",
    "social security number",
    "ssn",
    "pay before you start",
    "send money",
    "western union",
    "wire transfer",
    "cryptocurrency payment",
    "gift card",
]


FREE_WEBSITE_HOSTS = [
    "wixsite.com",
    "blogspot.com",
    "weebly.com",
    "wordpress.com",
    "sites.google.com",
    "carrd.co",
    "godaddysites.com",
]


_MULTI_PART_TLDS = {
    "co.uk",
    "org.uk",
    "ac.uk",
    "gov.uk",
    "co.in",
    "com.pk",
    "com.au",
    "co.jp",
    "com.br",
    "co.nz",
    "co.za",
    "com.sg",
    "com.hk",
}


# --------------------------------------------------------------------------
# Safe Streamlit secrets
# --------------------------------------------------------------------------

def get_secret(name: str, default=""):
    """
    Safely read a Streamlit secret.

    Important:
    st.secrets.get(...) can itself raise StreamlitSecretNotFoundError
    when secrets.toml does not exist.
    """
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# --------------------------------------------------------------------------
# Domain helpers
# --------------------------------------------------------------------------

def _root_domain(netloc: str) -> str:
    """Return a simple registrable/root domain."""
    netloc = (netloc or "").lower().split(":")[0]

    if netloc.startswith("www."):
        netloc = netloc[4:]

    parts = netloc.split(".")

    if len(parts) <= 2:
        return netloc

    last_two = ".".join(parts[-2:])

    if last_two in _MULTI_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])

    return last_two


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


# --------------------------------------------------------------------------
# Playwright / OCR readiness
# --------------------------------------------------------------------------

def ensure_chromium_installed() -> bool:
    """
    Try to make sure Playwright Chromium exists.

    On Streamlit Cloud, packages.txt is still recommended for OS libraries.
    """
    marker = "/tmp/.jabfraud_chromium_ready"

    if os.path.exists(marker):
        return True

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "playwright",
                "install",
                "chromium",
            ],
            capture_output=True,
            text=True,
            timeout=240,
        )

        if result.returncode == 0:
            try:
                with open(marker, "w") as f:
                    f.write("ok")
            except Exception:
                pass

            return True

        return False

    except Exception:
        return False


# --------------------------------------------------------------------------
# Stage 1 — Reading the job page
# --------------------------------------------------------------------------

def fetch_direct(url: str) -> dict:
    """
    Direct HTTP extraction without JavaScript.

    Retries transient failures (timeouts, connection resets, and 502/503/504
    responses) a couple of times with a short backoff before giving up, and
    rotates the User-Agent on each attempt. This is a best-effort measure —
    it will not get past real bot-protection, only real slowness/blips.
    """

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
        "attempts": 0,
    }

    max_attempts = 3
    retryable_status = {502, 503, 504}

    last_error = None

    for attempt in range(1, max_attempts + 1):

        out["attempts"] = attempt

        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": _random_user_agent(),
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,*/*;q=0.8"
                    ),
                },
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )

            out["status_code"] = resp.status_code
            out["final_url"] = resp.url
            out["redirect_chain"] = [r.url for r in resp.history]

            if resp.status_code in retryable_status and attempt < max_attempts:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(0.6 * attempt)
                continue

            if resp.status_code >= 400:
                out["error"] = f"HTTP {resp.status_code}"
                return out

            html = resp.text
            out["html"] = html

            extracted_text = trafilatura.extract(
                html,
                url=resp.url,
            ) or ""

            out["text"] = extracted_text.strip()

            if EXTRUCT_AVAILABLE:
                try:
                    base_url = get_base_url(
                        html,
                        resp.url,
                    )

                    data = extruct.extract(
                        html,
                        base_url=base_url,
                        syntaxes=["json-ld", "microdata"],
                    )

                    out["structured"] = data

                except Exception:
                    out["structured"] = {}

            has_structured_job = (
                _find_jobposting(out["structured"]) is not None
            )

            # We deliberately require meaningful content.
            out["success"] = (
                len(out["text"]) > 200
                or has_structured_job
            )

            return out

        except requests.RequestException as e:
            last_error = str(e)

            if attempt < max_attempts:
                time.sleep(0.6 * attempt)
                continue

            out["error"] = last_error
            return out

        except Exception as e:
            out["error"] = str(e)
            return out

    out["error"] = last_error or "Request failed after retries."
    return out


def fetch_with_browser(url: str) -> dict:
    """Browser extraction for JavaScript-heavy or blocked pages."""

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
            "Headless browser isn't available in this environment. "
            "Chromium or required OS libraries are missing. "
            "Add packages.txt and install Playwright dependencies."
        )

        return out

    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        out["error"] = f"Playwright not importable: {e}"
        return out

    try:
        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            page = browser.new_page(
                user_agent=_random_user_agent(),
                viewport={
                    "width": 1366,
                    "height": 900,
                },
                locale="en-US",
            )

            page.goto(
                url,
                timeout=30000,
                wait_until="domcontentloaded",
            )

            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=8000,
                )
            except Exception:
                pass

            # Give JS content some time.
            page.wait_for_timeout(1500)

            # Scroll for lazy-loaded content.
            for _ in range(5):
                try:
                    page.mouse.wheel(0, 1500)
                    page.wait_for_timeout(500)
                except Exception:
                    pass

            out["final_url"] = page.url

            html = page.content()
            out["html"] = html

            screenshot_bytes = page.screenshot(
                full_page=True
            )

            out["screenshot"] = screenshot_bytes

            browser.close()

        extracted_text = trafilatura.extract(
            html,
            url=out["final_url"],
        ) or ""

        out["text"] = extracted_text.strip()

        if EXTRUCT_AVAILABLE:
            try:
                base_url = get_base_url(
                    html,
                    out["final_url"],
                )

                out["structured"] = extruct.extract(
                    html,
                    base_url=base_url,
                    syntaxes=["json-ld", "microdata"],
                )

            except Exception:
                out["structured"] = {}

        has_structured_job = (
            _find_jobposting(out["structured"]) is not None
        )

        out["success"] = (
            len(out["text"]) > 200
            or has_structured_job
        )

        # OCR fallback.
        if (
            not out["success"]
            and OCR_AVAILABLE
            and out["screenshot"]
        ):
            try:
                img = Image.open(
                    io.BytesIO(out["screenshot"])
                )

                ocr_text = pytesseract.image_to_string(
                    img
                )

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
# Stage 2 — Structured data
# --------------------------------------------------------------------------

def _find_jobposting(structured: dict):
    """Find Schema.org JobPosting data."""

    if not structured:
        return None

    for block in structured.get("json-ld", []) or []:

        if not isinstance(block, dict):
            continue

        t = block.get("@type")

        if (
            t == "JobPosting"
            or (
                isinstance(t, list)
                and "JobPosting" in t
            )
        ):
            return block

        for node in block.get("@graph", []) or []:

            if not isinstance(node, dict):
                continue

            nt = node.get("@type")

            if (
                nt == "JobPosting"
                or (
                    isinstance(nt, list)
                    and "JobPosting" in nt
                )
            ):
                return node

    for block in structured.get("microdata", []) or []:

        if not isinstance(block, dict):
            continue

        if block.get("type", "").endswith(
            "JobPosting"
        ):
            return block.get("properties", {})

    return None


def _text_of(value):
    """Convert structured Schema.org values into text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return re.sub(
            r"<[^>]+>",
            " ",
            value,
        ).strip()

    if isinstance(value, dict):

        for key in (
            "name",
            "value",
            "@value",
            "streetAddress",
            "addressLocality",
        ):
            if key in value:
                return _text_of(value[key])

        return " ".join(
            _text_of(v)
            for v in value.values()
            if isinstance(v, (str, dict))
        )

    if isinstance(value, list):
        return ", ".join(
            _text_of(v)
            for v in value
        )

    return str(value)


# --------------------------------------------------------------------------
# Regex
# --------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+"
    r"@[a-zA-Z0-9.-]+\."
    r"[a-zA-Z]{2,}"
)


PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?"
    r"(?:\(?\d{2,4}\)?[\s.-]?){2,4}"
    r"\d{3,4}"
)


SALARY_RE = re.compile(
    r"(?:\$|₹|£|€|\bUSD\b|\bPKR\b|\bRs\.?\b)"
    r"\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:-|to)\s*"
    r"(?:\$|₹|£|€|\bUSD\b|\bPKR\b|\bRs\.?\b)?"
    r"\s?\d[\d,]*)?"
    r"\s*(?:/\s?"
    r"(?:year|yr|month|mo|hour|hr))?",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Job page detection
# --------------------------------------------------------------------------

def detect_page_type(
    soup,
    structured,
    text,
    url,
):
    """
    Determine whether this is likely:
      - a single job page
      - a listing page
      - blocked/limited content
    """

    job = _find_jobposting(structured)

    if job:
        return "single"

    path = urlparse(url).path.lower()

    single_job_patterns = [
        r"/jobs/view/",
        r"/job/",
        r"/jobs/",
        r"/job-detail/",
        r"/jobdetail/",
        r"/positions/",
        r"/vacancy/",
        r"/careers/.*\d+",
        r"/requisitions/",
        r"/apply/",
    ]

    listing_patterns = [
        r"/jobs/?$",
        r"/jobs/search",
        r"/job-search",
        r"/search/jobs",
        r"/careers/?$",
        r"/careers/search",
        r"/search",
    ]

    for pattern in single_job_patterns:
        if re.search(pattern, path):
            single_url = True
            break
    else:
        single_url = False

    for pattern in listing_patterns:
        if re.search(pattern, path):
            listing_url = True
            break
    else:
        listing_url = False

    job_title_indicators = [
        "software engineer",
        "software developer",
        "data scientist",
        "data analyst",
        "product manager",
        "project manager",
        "intern",
        "developer",
        "engineer",
        "designer",
        "accountant",
        "marketing",
        "sales",
        "analyst",
        "specialist",
        "administrator",
        "consultant",
        "manager",
    ]

    text_lower = (text or "").lower()

    title_indicator_found = any(
        phrase in text_lower
        for phrase in job_title_indicators
    )

    # Count job-looking links.
    job_link_count = 0

    if soup:
        hrefs = []

        for a in soup.find_all(
            "a",
            href=True,
        ):
            hrefs.append(a["href"])

        for href in hrefs:

            href_lower = href.lower()

            if (
                "/jobs/view/" in href_lower
                or "/job/" in href_lower
                or "/jobs/" in href_lower
                or "/jobdetail/" in href_lower
                or "/job-detail/" in href_lower
            ):
                job_link_count += 1

    if listing_url:
        return "listing"

    if job_link_count >= 8 and not single_url:
        return "listing"

    if single_url:
        return "single"

    if title_indicator_found and len(text) > 500:
        return "single"

    if len(text) < 150:
        return "limited"

    return "unknown"


# --------------------------------------------------------------------------
# Stage 2 — Parse job fields
# --------------------------------------------------------------------------

def parse_job_fields(
    fetch_result: dict,
    original_url: str,
) -> dict:

    html = fetch_result.get(
        "html",
        "",
    )

    text = (
        fetch_result.get("text", "")
        or fetch_result.get("ocr_text", "")
    )

    structured = fetch_result.get(
        "structured",
        {},
    )

    final_url = fetch_result.get(
        "final_url",
        original_url,
    )

    soup = (
        BeautifulSoup(
            html,
            "lxml",
        )
        if html
        else None
    )

    job = (
        _find_jobposting(structured)
        or {}
    )

    def sd(*keys, default=""):
        node = job

        for k in keys:

            if (
                isinstance(node, dict)
                and k in node
            ):
                node = node[k]
            else:
                return default

        return (
            _text_of(node)
            or default
        )

    # ------------------------------------------------------------------
    # Core fields
    # ------------------------------------------------------------------

    title = sd("title")

    if not title and soup and soup.title:
        title = soup.title.get_text(
            " ",
            strip=True,
        )

    company = sd(
        "hiringOrganization",
        "name",
    )

    location = (
        sd(
            "jobLocation",
            "address",
        )
        or sd("jobLocation")
    )

    salary = (
        sd(
            "baseSalary",
            "value",
            "value",
        )
        or sd("baseSalary")
    )

    employment_type = sd(
        "employmentType"
    )

    description_html = (
        job.get("description", "")
        if isinstance(job, dict)
        else ""
    )

    description = (
        _text_of(description_html)
        if description_html
        else ""
    )

    if not description:
        description = text[:6000]

    # ------------------------------------------------------------------
    # Salary
    # ------------------------------------------------------------------

    if not salary:
        m = SALARY_RE.search(text)

        if m:
            salary = m.group(0).strip()

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    if not location and soup:

        loc_meta = soup.find(
            "meta",
            attrs={
                "name": "job_location"
            },
        )

        if (
            loc_meta
            and loc_meta.get("content")
        ):
            location = loc_meta[
                "content"
            ]

    # Many ATS platforms (Greenhouse in particular) put the location in
    # the og:description meta tag as a short "City, Country" string,
    # since there's no separate structured location field on the page.
    if not location and soup:

        og_desc = soup.find(
            "meta",
            property="og:description",
        )

        if (
            og_desc
            and og_desc.get("content")
        ):
            candidate = og_desc["content"].strip()

            if (
                candidate
                and len(candidate) < 60
                and "," in candidate
                and not any(ch.isdigit() for ch in candidate)
            ):
                location = candidate

    # Last resort: scan the first few lines of visible text for something
    # that looks like a bare "City, Country" line — common right under the
    # job title on pages that don't expose location any other way.
    if not location and text:

        candidate_lines = [
            ln.strip()
            for ln in text.split("\n")[:6]
            if ln.strip()
        ]

        for line in candidate_lines:

            words = [
                w for w in re.split(r"[,\s]+", line) if w
            ]

            looks_like_location = (
                0 < len(line) < 60
                and "," in line
                and not any(ch.isdigit() for ch in line)
                and words
                and all(w[0].isupper() for w in words)
            )

            if looks_like_location:
                location = line
                break

    # ------------------------------------------------------------------
    # Company extraction
    # ------------------------------------------------------------------

    if not company and soup:

        og_site = soup.find(
            "meta",
            property="og:site_name",
        )

        if (
            og_site
            and og_site.get("content")
        ):
            candidate = og_site[
                "content"
            ].strip()

            # Don't use generic job-board names
            # as the employer.
            generic_domains = {
                "linkedin",
                "indeed",
                "glassdoor",
                "ziprecruiter",
                "monster",
                "google",
            }

            if candidate.lower() not in generic_domains:
                company = candidate

    # Logo alt text.
    if not company and soup:

        logo_img = soup.find(
            "img",
            alt=re.compile(
                r"logo",
                re.IGNORECASE,
            ),
        )

        if (
            logo_img
            and logo_img.get("alt")
        ):
            m = re.match(
                r"^(.*?)\s*logo\s*$",
                logo_img["alt"].strip(),
                re.IGNORECASE,
            )

            if m and m.group(1):
                candidate = m.group(1).strip()

                if candidate.lower() not in {
                    "linkedin",
                    "indeed",
                    "glassdoor",
                }:
                    company = candidate

    # Common ATS title pattern.
    if not company and soup:

        page_title = (
            soup.title.get_text(
                " ",
                strip=True,
            )
            if soup.title
            else ""
        )

        patterns = [
            r"\bat\s+(.+?)\s*$",
            r"\|\s*(.+?)\s*$",
            r"-\s*(.+?)\s*$",
        ]

        for pattern in patterns:

            m = re.search(
                pattern,
                page_title,
            )

            if m:
                candidate = m.group(1).strip()

                if (
                    candidate
                    and candidate.lower()
                    not in {
                        "linkedin",
                        "indeed",
                        "glassdoor",
                        "jobs",
                    }
                ):
                    company = candidate
                    break

    # ------------------------------------------------------------------
    # Detect page type
    # ------------------------------------------------------------------

    page_type = detect_page_type(
        soup,
        structured,
        text,
        final_url,
    )

    # Important:
    # Do NOT call linkedin.com the employer simply because
    # the employer couldn't be extracted.
    #
    # We use "Unknown" instead.
    if not company:
        company = "Unknown"

    # ------------------------------------------------------------------
    # Contact information
    # ------------------------------------------------------------------

    emails = sorted(
        set(
            EMAIL_RE.findall(
                text
            )
        )
    )

    recruiter_email = (
        emails[0]
        if emails
        else ""
    )

    phone_matches = PHONE_RE.findall(
        text
    )

    phones = []

    for m in phone_matches:

        digits = re.sub(
            r"\D",
            "",
            m,
        )

        if len(digits) >= 7:
            phones.append(m.strip())

    phones = sorted(set(phones))

    recruiter_phone = (
        phones[0]
        if phones
        else ""
    )

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    application_url = ""

    company_url = sd(
        "hiringOrganization",
        "sameAs",
    )

    external_links = []

    if soup:

        for a in soup.find_all(
            "a",
            href=True,
        ):

            href = a["href"]

            label = (
                a.get_text(
                    " ",
                    strip=True,
                )
                .lower()
            )

            full = urljoin(
                final_url,
                href,
            )

            # Application URL.
            if (
                not application_url
                and (
                    "apply" in label
                    or "apply" in href.lower()
                )
            ):
                application_url = full

            netloc = urlparse(
                full
            ).netloc

            page_netloc = urlparse(
                final_url
            ).netloc

            if (
                netloc
                and netloc != page_netloc
                and full.startswith(
                    "http"
                )
            ):
                external_links.append(
                    full
                )

    external_links = sorted(
        set(external_links)
    )[:15]

    domain = _domain_from_url(
        final_url
    )

    # ------------------------------------------------------------------
    # Experience
    # ------------------------------------------------------------------

    exp_match = re.search(
        r"\d+\+?(?:\s*[-–to]+\s*\d+\+?)?\s*years?\b",
        description,
        re.IGNORECASE,
    )

    experience = (
        exp_match.group(0)
        if exp_match
        else ""
    )

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    skill_keywords = [
        "python",
        "java",
        "javascript",
        "sql",
        "excel",
        "communication",
        "react",
        "node",
        "aws",
        "marketing",
        "sales",
        "customer service",
        "project management",
        "accounting",
        "design",
        "figma",
        "machine learning",
        "artificial intelligence",
        "ai",
        "docker",
        "kubernetes",
        "git",
        "linux",
    ]

    # Word-boundary matching, not plain substring: a plain "in" check would
    # match "ai" inside "trading" or "git" inside "digital", which produced
    # false-positive skills on real postings.
    description_lower = description.lower()

    skills_found = sorted(
        {
            kw
            for kw in skill_keywords
            if re.search(
                r"\b" + re.escape(kw) + r"\b",
                description_lower,
            )
        }
    )

    # ------------------------------------------------------------------
    # Benefits
    # ------------------------------------------------------------------

    benefit_keywords = [
        "health insurance",
        "paid leave",
        "remote work",
        "work from home",
        "bonus",
        "401k",
        "pension",
        "flexible hours",
        "medical",
        "paid time off",
        "life insurance",
    ]

    benefits_found = sorted(
        {
            kw
            for kw in benefit_keywords
            if re.search(
                r"\b" + re.escape(kw) + r"\b",
                description_lower,
            )
        }
    )

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
        "redirect_chain": fetch_result.get(
            "redirect_chain",
            [],
        ),
        "external_links": external_links,
        "raw_text": text,
        "screenshot": fetch_result.get(
            "screenshot"
        ),
        "page_type": page_type,
        "looks_like_listing_page": (
            page_type == "listing"
        ),
    }


# --------------------------------------------------------------------------
# Extraction confidence
# --------------------------------------------------------------------------

def compute_extraction_confidence(
    method: str,
    ocr_used: bool,
    job: dict,
) -> int:

    if ocr_used:
        base = 72
    elif method == "browser":
        base = 84
    else:
        base = 92

    important = [
        job.get("title", ""),
        job.get("company", ""),
        job.get("description", ""),
    ]

    missing = sum(
        1
        for f in important
        if not f
        or f == "Unknown"
    )

    base -= missing * 12

    if job.get("page_type") == "listing":
        base = min(
            base,
            35,
        )

    elif job.get("page_type") == "limited":
        base = min(
            base,
            45,
        )

    return max(
        20,
        min(97, base),
    )


# --------------------------------------------------------------------------
# Stage 3a — Rule-based fraud checklist
# --------------------------------------------------------------------------

def run_rule_checklist(
    job: dict,
    history: list = None,
) -> tuple:

    """
    Returns:
        rule_score 0-100
        flags (each with a "category" and "severity" for grouped display)

    `history` is the optional in-session list of previously analyzed jobs
    (see the sidebar history feature). When provided, a recruiter email or
    application domain that has already shown up under a *different*
    company name in this session is flagged as reused.
    """

    flags = []
    score = 0

    text_lower = (
        job.get("description", "")
        + " "
        + job.get("raw_text", "")
    ).lower()

    def add_flag(icon, title, detail, category, points):
        severity = (
            "high" if points >= 25
            else "medium" if points >= 12
            else "low"
        )

        flags.append(
            {
                "icon": icon,
                "title": title,
                "detail": detail,
                "category": category,
                "severity": severity,
            }
        )

    # ------------------------------------------------------------------
    # Page/extraction quality
    # ------------------------------------------------------------------

    if job.get("looks_like_listing_page"):

        add_flag(
            "⚠",
            "Job listing page detected",
            (
                "This appears to contain multiple jobs rather "
                "than one specific posting. Fraud analysis may "
                "be unreliable until a specific job URL is used."
            ),
            "Page Quality",
            5,
        )

        score += 5

    # ------------------------------------------------------------------
    # Company
    # ------------------------------------------------------------------

    company = (
        job.get("company", "")
        .strip()
    )

    if not company or company.lower() == "unknown":

        add_flag(
            "⚠",
            "Employer could not be verified",
            (
                "The page did not provide a clearly identifiable "
                "employer name."
            ),
            "Company",
            8,
        )

        score += 8

    # ------------------------------------------------------------------
    # Free company website
    # ------------------------------------------------------------------

    if job.get("company_url"):

        host = _domain_from_url(
            job["company_url"]
        )

        if any(
            free in host
            for free in FREE_WEBSITE_HOSTS
        ):

            add_flag(
                "⚠",
                "Company site uses a free website builder",
                (
                    f"The listed company website ({host}) "
                    "uses a free website-builder domain."
                ),
                "Company",
                12,
            )

            score += 12

    # ------------------------------------------------------------------
    # Recruiter email
    # ------------------------------------------------------------------

    email = job.get(
        "recruiter_email",
        "",
    )

    if email:

        email_domain = (
            email.split("@")[-1]
            .lower()
        )

        if (
            email_domain
            in FREE_EMAIL_DOMAINS
        ):

            add_flag(
                "⚠",
                "Personal email address",
                (
                    f"The recruiter used a personal "
                    f"email provider ({email_domain}) "
                    "instead of a company email."
                ),
                "Contact",
                18,
            )

            score += 18

        else:

            company_host = _domain_from_url(
                job.get(
                    "company_url",
                    "",
                )
            )

            email_root = _root_domain(
                email_domain
            )

            company_root = _root_domain(
                company_host
            )

            # Only flag this when there is enough information.
            if (
                email_root
                and company_root
                and email_root != company_root
            ):

                add_flag(
                    "⚠",
                    "Recruiter email does not match company website",
                    (
                        f"The recruiter email uses "
                        f"{email_domain}, while the company "
                        f"website uses {company_host}."
                    ),
                    "Contact",
                    12,
                )

                score += 12

        # Cross-check against this session's history: same recruiter
        # contact, different employer named across checks.
        if history:

            for past in history:

                past_email = (
                    past.get("recruiter_email", "")
                )

                past_company = (
                    past.get("company", "")
                )

                if (
                    past_email
                    and past_email.lower() == email.lower()
                    and past_company
                    and past_company.lower() != company.lower()
                ):

                    add_flag(
                        "🚨",
                        "Same recruiter contact used for a different company",
                        (
                            f"The email {email} was already seen in this "
                            f"session under a different employer name "
                            f"('{past_company}')."
                        ),
                        "Contact",
                        25,
                    )

                    score += 25
                    break

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    description_length = len(
        job.get(
            "description",
            "",
        )
    )

    if (
        description_length > 0
        and description_length < 150
    ):

        add_flag(
            "⚠",
            "Very short job description",
            (
                "The extracted posting contains very little "
                "information about the actual role."
            ),
            "Description",
            10,
        )

        score += 10

    # ------------------------------------------------------------------
    # Urgency
    # ------------------------------------------------------------------

    urgent_hits = [
        phrase
        for phrase in URGENCY_PHRASES
        if phrase in text_lower
    ]

    if urgent_hits:

        add_flag(
            "⚠",
            "Urgency language",
            (
                "The posting contains language that pressures "
                "applicants to act immediately."
            ),
            "Description",
            10,
        )

        score += 10

    # ------------------------------------------------------------------
    # Payment / sensitive information
    # ------------------------------------------------------------------

    payment_hits = [
        phrase
        for phrase in PAYMENT_RED_FLAG_PHRASES
        if phrase in text_lower
    ]

    if payment_hits:

        add_flag(
            "🚨",
            "Payment or sensitive-data request",
            (
                f"The posting contains language associated "
                f"with payment or sensitive information requests "
                f"('{payment_hits[0]}')."
            ),
            "Description",
            35,
        )

        score += 35

    # ------------------------------------------------------------------
    # Unrealistic salary
    # ------------------------------------------------------------------

    if (
        "no experience required" in text_lower
        or "no experience needed" in text_lower
    ):

        salary_text = job.get(
            "salary",
            "",
        )

        salary_digits = re.sub(
            r"[^\d]",
            "",
            salary_text,
        )

        if salary_digits:

            try:
                salary_number = int(
                    salary_digits[:6]
                )

                if salary_number > 50000:

                    add_flag(
                        "⚠",
                        "Potentially unrealistic salary",
                        (
                            "The posting combines no-experience "
                            "requirements with a potentially high "
                            "salary."
                        ),
                        "Description",
                        15,
                    )

                    score += 15

            except Exception:
                pass

    # ------------------------------------------------------------------
    # Application URL
    # ------------------------------------------------------------------

    if job.get("application_url"):

        apply_host = _domain_from_url(
            job["application_url"]
        )

        page_host = (
            job.get(
                "page_domain",
                "",
            )
            .lower()
        )

        company_host = _domain_from_url(
            job.get(
                "company_url",
                "",
            )
        )

        apply_root = _root_domain(
            apply_host
        )

        page_root = _root_domain(
            page_host
        )

        company_root = _root_domain(
            company_host
        )

        known_ats_hosts = (
            "greenhouse.io",
            "lever.co",
            "myworkdayjobs.com",
            "icims.com",
            "smartrecruiters.com",
            "bamboohr.com",
            "breezy.hr",
            "ashbyhq.com",
            "jobvite.com",
            "taleo.net",
            "workable.com",
        )

        is_known_ats = any(
            apply_root.endswith(host)
            for host in known_ats_hosts
        )

        # External applications are NOT automatically fraud.
        # Only flag if it is an unrelated domain and not a known ATS.
        if (
            apply_root
            and page_root
            and apply_root != page_root
            and apply_root != company_root
            and not is_known_ats
        ):

            add_flag(
                "⚠",
                "Unverified external application link",
                (
                    f"The Apply link leads to {apply_host}, "
                    "which could not be matched to the employer "
                    "or a recognized ATS."
                ),
                "Application",
                10,
            )

            score += 10

    return (
        min(100, score),
        flags,
    )


# --------------------------------------------------------------------------
# Stage 3b — AI analysis
# --------------------------------------------------------------------------

def get_groq_client():

    api_key = (
        st.session_state.get(
            "groq_api_key",
            ""
        )
        or get_secret(
            "GROQ_API_KEY",
            ""
        )
        or get_secret(
            "GROK_API_KEY",
            ""
        )
    )

    if (
        not api_key
        or not OPENAI_SDK_AVAILABLE
    ):
        return None

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


AI_SYSTEM_PROMPT = """
You are a careful job-fraud review assistant.

Your job is to evaluate whether a job posting contains genuine scam/fraud
signals.

IMPORTANT RULES:

1. Do NOT assume a job is fraudulent simply because it is posted on LinkedIn,
   Indeed, Glassdoor, or another job board.

2. Do NOT treat missing salary as evidence of fraud.

3. Do NOT treat missing location as evidence of fraud.

4. Do NOT treat a third-party ATS such as Greenhouse, Lever, Workday,
   SmartRecruiters, Ashby, or similar as suspicious by itself.

5. A job-board domain is NOT the employer. If the employer cannot be extracted,
   say that employer verification is incomplete instead of claiming the
   job-board company is the employer.

6. Strong fraud indicators include:
   - requests for money
   - registration fees
   - fake checks
   - cryptocurrency payments
   - gift cards
   - requests for bank credentials before hiring
   - requests for highly sensitive information before a legitimate hiring step
   - suspicious external application domains
   - impersonation
   - major contradictions
   - unrealistic promises
   - strong pressure tactics

7. Missing information should reduce confidence in the analysis rather than
   automatically increase the fraud score.

8. Consider the extraction quality. If the page was partially blocked or
   only a listing page was extracted, lower confidence and explain that.

9. You will also be given a "rule_based_flags" list, produced by a separate
   deterministic checklist that already ran on this same posting. Treat it
   as a second opinion, not ground truth: explicitly agree or disagree with
   each one you find material, and explain why in your findings. Do not
   just restate it — add your own independent read of the text.

10. Report how confident you are in your own risk_score, given how complete
    and reliable the extracted data looked.

Return ONLY one JSON object:

{
  "risk_score": integer 0-100,
  "ai_confidence": integer 0-100,
  "findings": [
    {
      "issue": "short title",
      "explanation": "one plain-English sentence"
    }
  ],
  "recommendation": "one short plain-English sentence"
}
"""


def run_ai_analysis(
    job: dict,
    rule_flags: list = None,
) -> dict:

    client = get_groq_client()

    if client is None:

        return {
            "risk_score": None,
            "ai_confidence": None,
            "findings": [],
            "recommendation": "",
            "error": (
                "No Groq API key configured. "
                "Add GROQ_API_KEY in Streamlit secrets "
                "or enter it in the sidebar."
            ),
        }

    model = get_secret(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL,
    )

    rule_flags_summary = [
        {
            "title": f.get("title"),
            "category": f.get("category"),
            "severity": f.get("severity"),
        }
        for f in (rule_flags or [])
    ]

    payload = {
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "salary": job.get("salary"),
        "employment_type": job.get(
            "employment_type"
        ),
        "description": (
            job.get(
                "description",
                "",
            )[:4000]
        ),
        "recruiter_email": job.get(
            "recruiter_email"
        ),
        "recruiter_phone": job.get(
            "recruiter_phone"
        ),
        "application_url": job.get(
            "application_url"
        ),
        "company_url": job.get(
            "company_url"
        ),
        "page_domain": job.get(
            "page_domain"
        ),
        "page_type": job.get(
            "page_type"
        ),
        "external_links": job.get(
            "external_links"
        ),
        "rule_based_flags": rule_flags_summary,
    }

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": AI_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload
                    ),
                },
            ],
            temperature=0.2,
            response_format={
                "type": "json_object"
            },
            timeout=30,
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        parsed = json.loads(
            content
        )

        risk_score = int(
            parsed.get(
                "risk_score",
                0,
            )
        )

        risk_score = max(
            0,
            min(
                100,
                risk_score,
            ),
        )

        ai_confidence = parsed.get(
            "ai_confidence",
            None,
        )

        if ai_confidence is not None:

            try:
                ai_confidence = max(
                    0,
                    min(
                        100,
                        int(ai_confidence),
                    ),
                )
            except Exception:
                ai_confidence = None

        findings = (
            parsed.get(
                "findings",
                [],
            )
            or []
        )

        recommendation = (
            parsed.get(
                "recommendation",
                "",
            )
            or ""
        )

        return {
            "risk_score": risk_score,
            "ai_confidence": ai_confidence,
            "findings": findings,
            "recommendation": recommendation,
            "error": None,
        }

    except json.JSONDecodeError:

        return {
            "risk_score": None,
            "ai_confidence": None,
            "findings": [],
            "recommendation": "",
            "error": (
                "Groq responded, but not with valid JSON."
            ),
        }

    except Exception as e:

        try:

            import openai as _openai_mod

            if isinstance(
                e,
                _openai_mod.AuthenticationError,
            ):

                msg = (
                    "Authentication failed — "
                    "check that the Groq API key is correct."
                )

            elif isinstance(
                e,
                _openai_mod.NotFoundError,
            ):

                msg = (
                    f"Model '{model}' was not found by Groq. "
                    "Check GROQ_MODEL."
                )

            elif isinstance(
                e,
                _openai_mod.RateLimitError,
            ):

                msg = (
                    "Groq rate limit reached or quota exceeded."
                )

            elif isinstance(
                e,
                _openai_mod.APIConnectionError,
            ):

                msg = (
                    "Couldn't reach api.groq.com. "
                    "Check internet/network access."
                )

            elif isinstance(
                e,
                _openai_mod.APIStatusError,
            ):

                msg = (
                    f"Groq API returned HTTP "
                    f"{e.status_code}: "
                    f"{getattr(e, 'message', str(e))}"
                )

            else:
                msg = str(e)

        except Exception:
            msg = str(e)

        return {
            "risk_score": None,
            "ai_confidence": None,
            "findings": [],
            "recommendation": "",
            "error": msg,
        }


# --------------------------------------------------------------------------
# Stage 4 — Combined risk score
# --------------------------------------------------------------------------

def combine_scores(
    rule_score: int,
    ai_score,
    ai_confidence=None,
    extraction_confidence: int = 100,
) -> int:

    if ai_score is None:
        return rule_score

    # Weight the AI's contribution by how confident it says it is. A model
    # that admits low confidence (e.g. because extraction was poor) should
    # not be allowed to swing the combined score as hard as a confident one.
    if ai_confidence is not None:
        ai_weight = 0.55 * (ai_confidence / 100)
        ai_weight = max(0.15, min(0.55, ai_weight))
    else:
        ai_weight = 0.55

    rule_weight = 1 - ai_weight

    score = round(
        rule_weight * rule_score
        + ai_weight * ai_score
    )

    # If extraction is very poor, do not allow the system to confidently
    # produce a very high fraud score based on incomplete information.
    if extraction_confidence < 45:
        score = min(
            score,
            55,
        )

    return max(
        0,
        min(
            100,
            score,
        ),
    )


def risk_label(
    score: int,
) -> tuple:

    if score <= 30:
        return (
            "🟢",
            "Low Risk",
            "#2ecc71",
        )

    elif score <= 60:
        return (
            "🟡",
            "Moderate Risk",
            "#f1c40f",
        )

    elif score <= 80:
        return (
            "🟠",
            "Suspicious",
            "#e67e22",
        )

    else:
        return (
            "🔴",
            "High Risk",
            "#e74c3c",
        )


SEVERITY_COLOR = {
    "high": "#e74c3c",
    "medium": "#e67e22",
    "low": "#7f8c8d",
}


FLAG_CATEGORY_ORDER = [
    "Page Quality",
    "Company",
    "Contact",
    "Description",
    "Application",
    "AI Analysis",
]


def render_score_gauge(
    score: int,
    color: str,
) -> str:
    """
    A small CSS conic-gradient arc gauge. Returns raw HTML — caller is
    responsible for passing it to st.markdown(..., unsafe_allow_html=True).
    """

    angle = round(
        (score / 100) * 360
    )

    return f"""
    <div style="
        width:180px;
        height:180px;
        border-radius:50%;
        margin:0 auto;
        background:conic-gradient(
            {color} 0deg {angle}deg,
            #2a2a2a {angle}deg 360deg
        );
        display:flex;
        align-items:center;
        justify-content:center;
    ">
        <div style="
            width:140px;
            height:140px;
            border-radius:50%;
            background:#0e1117;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
        ">
            <div style="font-size:38px;font-weight:700;color:{color}">
                {score}
            </div>
            <div style="font-size:12px;color:#9a9a9a">out of 100</div>
        </div>
    </div>
    """


def build_text_report(entry: dict) -> str:
    """Plain-text version of a result entry, for the download button."""

    job = entry["job"]
    ai_result = entry["ai_result"]

    lines = []

    lines.append("JabFraud — Fraud Risk Report")
    lines.append("=" * 32)
    lines.append(f"Checked: {entry.get('timestamp', '')}")
    lines.append(f"URL: {entry.get('url', '')}")
    lines.append("")
    lines.append(
        f"Final score: {entry['final_score']} / 100 "
        f"({entry['label']})"
    )
    lines.append(f"Rule-based score: {entry['rule_score']} / 100")

    if ai_result.get("risk_score") is not None:
        lines.append(
            f"AI score: {ai_result['risk_score']} / 100 "
            f"(AI confidence: {ai_result.get('ai_confidence', '—')})"
        )

    lines.append(
        f"Extraction confidence: {entry['confidence']}% "
        f"(method: {entry['method']})"
    )
    lines.append("")

    lines.append("Job Details")
    lines.append("-" * 32)
    lines.append(f"Title: {job.get('title') or '—'}")
    lines.append(f"Company: {job.get('company') or '—'}")
    lines.append(f"Location: {job.get('location') or '—'}")
    lines.append(f"Salary: {job.get('salary') or '—'}")
    lines.append(f"Recruiter email: {job.get('recruiter_email') or '—'}")
    lines.append(f"Recruiter phone: {job.get('recruiter_phone') or '—'}")
    lines.append(f"Application URL: {job.get('application_url') or '—'}")
    lines.append("")

    lines.append("Flags")
    lines.append("-" * 32)

    all_findings = entry.get("all_findings", [])

    if not all_findings:
        lines.append("No obvious warning signs were detected.")
    else:
        for f in all_findings:
            lines.append(
                f"[{f.get('category', 'General')}] "
                f"{f.get('title', '')} — {f.get('detail', '')}"
            )

    lines.append("")
    lines.append("Recommendation")
    lines.append("-" * 32)
    lines.append(entry.get("recommendation", ""))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(
    page_title=(
        "JabFraud — Check the job before you apply."
    ),
    page_icon="🕵️",
    layout="centered",
)


if "history" not in st.session_state:
    st.session_state["history"] = []

if "active" not in st.session_state:
    st.session_state["active"] = None

if "_fetch_cache" not in st.session_state:
    st.session_state["_fetch_cache"] = {}


FETCH_CACHE_TTL_SECONDS = 300


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:

    st.markdown("### Settings")

    secret_key_present = bool(
        get_secret(
            "GROQ_API_KEY",
            "",
        )
        or get_secret(
            "GROK_API_KEY",
            "",
        )
    )

    if secret_key_present:

        st.success(
            "Groq API key loaded from Streamlit secrets."
        )

    else:

        st.session_state[
            "groq_api_key"
        ] = st.text_input(
            "Groq API key (groq.com)",
            type="password",
            help=(
                "Not stored anywhere except this "
                "browser session."
            ),
        )

    st.caption(
        "Playwright/OCR fallback needs OS packages "
        "(Chromium libraries and tesseract-ocr). "
        "On Streamlit Community Cloud, add a "
        "packages.txt file. Without it, JabFraud "
        "still works for pages readable by direct extraction."
    )

    st.markdown("---")
    st.markdown("### History")

    if not st.session_state["history"]:

        st.caption(
            "Jobs you check this session will show up here."
        )

    else:

        for i, entry in enumerate(
            st.session_state["history"]
        ):

            label = (
                f"{entry['icon']} "
                f"{entry['job'].get('title') or entry['url']}"
                f" — {entry['final_score']}"
            )

            if st.button(
                label,
                key=f"history_{i}",
                use_container_width=True,
            ):

                st.session_state["active"] = entry
                st.rerun()


# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------

st.title("🕵️ JabFraud")

st.caption(
    f"**{APP_TAGLINE}**"
)


url = st.text_input(
    "Paste Job URL",
    placeholder=(
        "https://example.com/careers/job/12345"
    ),
)


analyze_clicked = st.button(
    "🔍 Analyze Job",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------------------------------
# Analyze
# --------------------------------------------------------------------------

if analyze_clicked:

    if (
        not url
        or not url.strip()
        .lower()
        .startswith(
            (
                "http://",
                "https://",
            )
        )
    ):

        st.error(
            "Please paste a full job posting URL, "
            "starting with http:// or https://"
        )

        st.stop()

    url = url.strip()

    progress = st.status(
        "Extracting job data…",
        expanded=True,
    )

    # ------------------------------------------------------------------
    # Stage 1 — Direct extraction (with a short in-session cache so
    # clicking Analyze again on the same URL within a few minutes
    # doesn't re-scrape it from scratch)
    # ------------------------------------------------------------------

    cache = st.session_state["_fetch_cache"]

    cached = cache.get(url)

    if (
        cached
        and time.time() - cached["cached_at"] < FETCH_CACHE_TTL_SECONDS
    ):

        progress.write(
            "Reusing a recent result for this exact URL…"
        )

        fetch_result = cached["fetch_result"]

    else:

        fetch_result = fetch_direct(
            url
        )

        cache[url] = {
            "cached_at": time.time(),
            "fetch_result": fetch_result,
        }

    ocr_used = False

    # First parse direct result to determine whether it is actually useful.
    direct_job = None

    if (
        fetch_result.get("success")
        or fetch_result.get("html")
    ):

        direct_job = parse_job_fields(
            fetch_result,
            url,
        )

    # ------------------------------------------------------------------
    # Decide whether browser fallback is necessary
    # ------------------------------------------------------------------

    browser_needed = False

    if not fetch_result.get(
        "success"
    ):

        browser_needed = True

    elif direct_job:

        direct_page_type = direct_job.get(
            "page_type"
        )

        direct_text_length = len(
            direct_job.get(
                "raw_text",
                "",
            )
        )

        # Browser fallback for:
        # - listing page
        # - limited extraction
        # - extremely short extraction
        # - unknown page where text is weak
        if direct_page_type in {
            "listing",
            "limited",
        }:
            browser_needed = True

        elif direct_text_length < 500:
            browser_needed = True

    if browser_needed:

        progress.write(
            "Direct extraction is incomplete or the page "
            "appears dynamic/list-based. Trying browser extraction…"
        )

        browser_result = fetch_with_browser(
            url
        )

        if (
            browser_result.get("success")
            or browser_result.get("html")
        ):

            browser_job = parse_job_fields(
                browser_result,
                url,
            )

            # Prefer browser extraction if:
            # - it found a single job
            # - it has more text
            # - it found structured data
            # - direct was only a listing
            browser_text_len = len(
                browser_job.get(
                    "raw_text",
                    "",
                )
            )

            direct_text_len = (
                len(
                    direct_job.get(
                        "raw_text",
                        "",
                    )
                )
                if direct_job
                else 0
            )

            browser_is_better = (
                browser_job.get(
                    "page_type"
                ) == "single"
                and (
                    not direct_job
                    or direct_job.get(
                        "page_type"
                    ) != "single"
                    or browser_text_len
                    > direct_text_len
                )
            )

            if browser_is_better:

                fetch_result = browser_result

                ocr_used = bool(
                    browser_result.get(
                        "ocr_text"
                    )
                    and not (
                        browser_result.get(
                            "text"
                        )
                        or ""
                    ).strip()
                )

                progress.write(
                    "Browser extraction succeeded."
                )

            elif (
                not fetch_result.get(
                    "success"
                )
                and browser_result.get(
                    "success"
                )
            ):

                fetch_result = browser_result

                ocr_used = bool(
                    browser_result.get(
                        "ocr_text"
                    )
                )

                progress.write(
                    "Browser extraction succeeded."
                )

            else:

                progress.write(
                    "Direct extraction contained more useful "
                    "content, so it was retained."
                )

        elif browser_result.get(
            "error"
        ):

            progress.write(
                "Browser fallback unavailable: "
                + browser_result[
                    "error"
                ]
            )

    # ------------------------------------------------------------------
    # Nothing readable
    # ------------------------------------------------------------------

    if (
        not fetch_result.get("success")
        and not fetch_result.get("html")
    ):

        progress.update(
            label="Could not read this page.",
            state="error",
        )

        st.error(
            "JabFraud couldn't extract enough content "
            "from this URL."
        )

        if fetch_result.get(
            "error"
        ):
            st.caption(
                "Details: "
                + fetch_result[
                    "error"
                ]
            )

        st.stop()

    # ------------------------------------------------------------------
    # Parse final job
    # ------------------------------------------------------------------

    progress.write(
        "Pulling out key job details…"
    )

    job = parse_job_fields(
        fetch_result,
        url,
    )

    confidence = compute_extraction_confidence(
        fetch_result.get(
            "method",
            "direct",
        ),
        ocr_used,
        job,
    )

    listing_warning = job.get(
        "looks_like_listing_page"
    )

    limited_warning = (
        job.get("page_type")
        == "limited"
    )

    if listing_warning:
        confidence = min(confidence, 40)

    if limited_warning:
        confidence = min(confidence, 45)

    # ------------------------------------------------------------------
    # Rule analysis (cross-checked against this session's history)
    # ------------------------------------------------------------------

    progress.write(
        "Checking against the fraud checklist…"
    )

    rule_score, rule_flags = run_rule_checklist(
        job,
        history=[
            e["job"] for e in st.session_state["history"]
        ],
    )

    # ------------------------------------------------------------------
    # AI analysis
    # ------------------------------------------------------------------

    progress.write(
        "Running AI fraud analysis…"
    )

    ai_result = run_ai_analysis(
        job,
        rule_flags=rule_flags,
    )

    # ------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------

    final_score = combine_scores(
        rule_score,
        ai_result.get("risk_score"),
        ai_result.get("ai_confidence"),
        confidence,
    )

    icon, label, color = risk_label(
        final_score
    )

    progress.update(
        label="Fraud analysis completed",
        state="complete",
        expanded=False,
    )

    # ------------------------------------------------------------------
    # Assemble the findings list (rule flags + AI findings, tagged)
    # ------------------------------------------------------------------

    all_findings = list(rule_flags)

    for f in ai_result.get("findings", []):

        all_findings.append(
            {
                "icon": "🤖",
                "title": f.get("issue", "Flag"),
                "detail": f.get("explanation", ""),
                "category": "AI Analysis",
                "severity": "medium",
            }
        )

    recommendation = (
        ai_result.get("recommendation")
        or (
            "Verify this position through the "
            "company's official careers page before "
            "providing personal information."
        )
    )

    # ------------------------------------------------------------------
    # Build the result entry, save to history, and make it active
    # ------------------------------------------------------------------

    entry = {
        "url": url,
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        ),
        "job": job,
        "rule_score": rule_score,
        "ai_result": ai_result,
        "final_score": final_score,
        "icon": icon,
        "label": label,
        "color": color,
        "confidence": confidence,
        "method": (
            fetch_result.get("method", "direct")
            + (" + OCR" if ocr_used else "")
        ),
        "all_findings": all_findings,
        "recommendation": recommendation,
        "listing_warning": listing_warning,
        "limited_warning": limited_warning,
    }

    # De-duplicate: if this URL is already in history, replace it and
    # move it to the top instead of growing the list forever.
    st.session_state["history"] = [
        e for e in st.session_state["history"]
        if e["url"] != url
    ]

    st.session_state["history"].insert(0, entry)

    # Keep the sidebar list from growing without bound.
    st.session_state["history"] = (
        st.session_state["history"][:20]
    )

    st.session_state["active"] = entry


# --------------------------------------------------------------------------
# Render whatever result is currently active (freshly analyzed, or picked
# from history in the sidebar)
# --------------------------------------------------------------------------

active = st.session_state.get("active")

if active is None:

    st.caption(
        "Paste a job posting link above and click "
        "**Analyze Job** to get a risk report."
    )

else:

    job = active["job"]
    ai_result = active["ai_result"]

    st.markdown("---")

    if active.get("listing_warning"):

        st.warning(
            "This looks like a job-board **listing page** "
            "rather than one specific posting. Results may "
            "be unreliable. Open one specific job and paste "
            "that URL instead."
        )

    if active.get("limited_warning"):

        st.warning(
            "Only limited job content could be extracted "
            "from this page. The browser/screenshot fallback "
            "may not be available on this deployment."
        )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.markdown(
            render_score_gauge(
                active["final_score"],
                active["color"],
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='text-align:center;font-size:18px;"
            f"font-weight:600;color:{active['color']}'>"
            f"{active['icon']} {active['label']}</div>",
            unsafe_allow_html=True,
        )

    with col2:

        st.metric(
            "Rule-based score",
            f"{active['rule_score']} / 100",
        )

        st.metric(
            "AI score",
            (
                f"{ai_result['risk_score']} / 100"
                if ai_result.get("risk_score") is not None
                else "—"
            ),
        )

        if ai_result.get("ai_confidence") is not None:

            st.caption(
                f"AI self-reported confidence: "
                f"{ai_result['ai_confidence']}%"
            )

        st.caption(
            f"Extraction confidence: "
            f"{active['confidence']}% "
            f"(method: {active['method']})"
        )

    if ai_result.get("error"):

        st.warning(
            "AI analysis unavailable:"
        )

        st.code(
            ai_result["error"],
            language=None,
        )

    # ------------------------------------------------------------------
    # Why this score — grouped by category, color-coded by severity
    # ------------------------------------------------------------------

    st.subheader("Why this score")

    all_findings = active.get(
        "all_findings",
        [],
    )

    if not all_findings:

        st.success(
            "No obvious warning signs were detected."
        )

    else:

        by_category = {}

        for f in all_findings:

            cat = f.get("category", "General")
            by_category.setdefault(cat, []).append(f)

        ordered_categories = [
            c for c in FLAG_CATEGORY_ORDER if c in by_category
        ] + [
            c for c in by_category if c not in FLAG_CATEGORY_ORDER
        ]

        for cat in ordered_categories:

            with st.expander(
                f"{cat} ({len(by_category[cat])})",
                expanded=True,
            ):

                for f in by_category[cat]:

                    sev_color = SEVERITY_COLOR.get(
                        f.get("severity", "medium"),
                        "#e67e22",
                    )

                    st.markdown(
                        f"<span style='color:{sev_color};"
                        f"font-weight:600'>{f['icon']} "
                        f"{f['title']}</span> — {f['detail']}",
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    st.info(
        f"**Recommendation:** {active['recommendation']}"
    )

    # ------------------------------------------------------------------
    # Download report
    # ------------------------------------------------------------------

    st.download_button(
        "⬇ Download report (.txt)",
        data=build_text_report(active),
        file_name="jabfraud_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # ------------------------------------------------------------------
    # Job details
    # ------------------------------------------------------------------

    st.subheader("Job Details")

    d1, d2 = st.columns(2)

    with d1:

        st.markdown(f"**Title:** {job['title'] or '—'}")
        st.markdown(f"**Company:** {job['company'] or '—'}")
        st.markdown(f"**Location:** {job['location'] or '—'}")
        st.markdown(f"**Salary:** {job['salary'] or '—'}")
        st.markdown(
            f"**Employment type:** "
            f"{job['employment_type'] or '—'}"
        )
        st.markdown(f"**Experience:** {job['experience'] or '—'}")

    with d2:

        st.markdown(
            f"**Recruiter email:** "
            f"{job['recruiter_email'] or '—'}"
        )
        st.markdown(
            f"**Recruiter phone:** "
            f"{job['recruiter_phone'] or '—'}"
        )
        st.markdown(
            f"**Application URL:** "
            f"{job['application_url'] or '—'}"
        )
        st.markdown(
            f"**Company URL:** "
            f"{job['company_url'] or '—'}"
        )
        st.markdown(
            f"**Page domain:** "
            f"{job['page_domain'] or '—'}"
        )
        st.markdown(
            f"**Page type:** "
            f"{job.get('page_type', 'unknown')}"
        )

        if job["redirect_chain"]:

            st.markdown(
                f"**Redirects:** "
                f"{' → '.join(job['redirect_chain'])}"
            )

    if job["skills"]:

        st.markdown(
            f"**Skills mentioned:** "
            f"{', '.join(job['skills'])}"
        )

    if job["benefits"]:

        st.markdown(
            f"**Benefits mentioned:** "
            f"{', '.join(job['benefits'])}"
        )

    if job["external_links"]:

        with st.expander(
            f"External links found on the page "
            f"({len(job['external_links'])})"
        ):

            for link in job["external_links"]:
                st.write(link)

    if job.get("screenshot"):

        st.subheader("Page Evidence")

        st.image(
            job["screenshot"],
            caption="Captured screenshot of the job page",
            use_container_width=True,
        )

    with st.expander("Raw extracted text"):

        st.text(
            job["raw_text"][:5000]
            or "No text extracted."
        )
