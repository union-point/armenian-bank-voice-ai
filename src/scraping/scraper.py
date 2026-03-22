import json
import logging
import os
import re
import sys
from typing import Any, Dict, List
from urllib.parse import urljoin

import requests
import trafilatura
import yaml
from bs4 import BeautifulSoup
from src.scraping.utils.pdf_utils import download_file, extract_text_from_pdf
from src.scraping.utils.text_utils import (
    clean_for_rag,
    clean_for_rag_aggressive,
    is_scrapable_text,
    normalize_text,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BankScraper:
    def __init__(self, config_path: str = "config/banks.yaml"):
        self.config_path = config_path
        self.banks_config = self.load_config()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "hy,en;q=0.9,ru;q=0.8",
        }

    def load_config(self) -> Dict[str, Any]:
        """Load banks configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.error(f"Config file not found: {self.config_path}")
            return {"banks": {}}

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def scrape_all_banks(self):
        """Iterate through all banks and categories to scrape data."""
        banks = self.banks_config.get("banks", {})
        for bank_name, config in banks.items():
            logger.info(f"Processing bank: {bank_name}")
            self.scrape_bank_category(
                bank_name, config, "credits", config.get("credits_urls", [])
            )
            self.scrape_bank_category(
                bank_name, config, "deposits", config.get("deposits_urls", [])
            )
            self.scrape_bank_branches(bank_name, config)

    def scrape_bank_category(
        self, bank_name: str, config: Dict[str, Any], category: str, urls: List[str]
    ):
        """Scrape credits or deposits for a bank."""
        if not urls:
            logger.warning(f"No URLs defined for {bank_name} - {category}")
            return

        results = []
        base_url = config.get("base_url", "")
        pdf_selector = config.get("pdf_link_selector", "a[href$='.pdf']")

        for url in urls:
            logger.info(f"Scraping {category} from {url}")
            try:
                response = requests.get(url, headers=self.headers, timeout=20)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Strategy 1: Find PDF links
                pdf_links = [
                    urljoin(base_url, a["href"])
                    for a in soup.select(pdf_selector)
                    if a.has_attr("href")
                ]
                # Deduplicate links
                pdf_links = list(set(pdf_links))

                pdf_content_found = False
                for pdf_url in pdf_links:
                    logger.debug(f"Parsing PDF: {pdf_url}")
                    pdf_content = download_file(pdf_url, headers=self.headers)
                    text = None
                    if pdf_content:
                        text = extract_text_from_pdf(pdf_content)
                    if text:
                        # Try regular cleaning first, then aggressive if too noisy
                        cleaned_text = clean_for_rag(text)
                        if cleaned_text and len(cleaned_text) > 50:
                            # Check if text still has too much noise
                            bullet_count = cleaned_text.count("•")
                            starts_with_digits = re.match(
                                r"^[\d\s\n]+$", cleaned_text[:100]
                            )
                            if bullet_count > 10 or starts_with_digits:
                                cleaned_text = clean_for_rag_aggressive(text)

                            # Final check: is this actually scrapable text?
                            if is_scrapable_text(cleaned_text):
                                results.append(
                                    {
                                        "text": cleaned_text,
                                        "metadata": {
                                            "category": category[
                                                :-1
                                            ],  # Singular "credit" or "deposit"
                                            "bank_name": bank_name,
                                            "source_url": pdf_url,
                                        },
                                    }
                                )
                            pdf_content_found = True

                # Fallback Strategy: Extract HTML content if no PDF content was extracted
                if not pdf_content_found:
                    logger.info(
                        f"No PDF text extracted for {url}. Falling back to HTML."
                    )
                    downloaded = trafilatura.fetch_url(url)
                    text = trafilatura.extract(downloaded)
                    if text:
                        results.append(
                            {
                                "text": normalize_text(text),
                                "metadata": {
                                    "category": category[:-1],
                                    "bank_name": bank_name,
                                    "source_url": url,
                                },
                            }
                        )
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")

        if results:
            self.save_results(bank_name, category, results)

    def scrape_bank_branches(self, bank_name: str, config: Dict[str, Any]):
        """Scrape branch locations for a bank."""
        url = config.get("branches_url")
        if not url:
            logger.warning(f"No branch URL defined for {bank_name}")
            return

        logger.info(f"Scraping branches from {url}")
        results = []
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded, include_tables=True)

            if text:
                results.append(
                    {
                        "text": normalize_text(text),
                        "metadata": {
                            "category": "branch",
                            "bank_name": bank_name,
                            "source_url": url,
                        },
                    }
                )
        except Exception as e:
            logger.error(f"Error scraping branches from {url}: {e}")

        if results:
            self.save_results(bank_name, "branches", results)

    def save_results(self, bank_name: str, category: str, data: List[Dict[str, Any]]):
        """Save results as JSON."""
        output_dir = os.path.join("data", bank_name)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{category}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Saved {len(data)} entries for {bank_name} - {category} to {output_path}"
        )


if __name__ == "__main__":
    scraper = BankScraper()
    scraper.scrape_all_banks()
