"""
Web scraper for competitor prices.
Uses BeautifulSoup with mock URLs for demo; falls back to mock API.
"""
import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scraping.mock_scraper import fetch_competitor_price as mock_fetch

logger = logging.getLogger("scraper")

# Mock URLs for demonstration — these simulate real e-commerce pages
MOCK_URLS = {
    "amazon":   "https://www.amazon.in/dp/{product_id}",
    "flipkart": "https://www.flipkart.com/product/{product_id}",
}


def scrape_price_from_url(url: str, product_id: str) -> dict | None:
    """
    Attempt to scrape price from a given URL.
    In production, this would parse real HTML. For demo, it falls back to mock.
    """
    try:
        logger.info(f"Attempting to scrape: {url}")
        response = requests.get(url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DynamicPricingBot/1.0"
        })

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Try common price selectors (would work on real sites)
            price_selectors = [
                {"class": "a-price-whole"},       # Amazon
                {"class": "_30jeq3"},             # Flipkart
                {"class": "price"},               # Generic
                {"itemprop": "price"},            # Schema.org
            ]
            for selector in price_selectors:
                element = soup.find("span", selector) or soup.find("div", selector)
                if element:
                    price_text = element.get_text(strip=True)
                    price_text = price_text.replace(",", "").replace("₹", "").replace("$", "")
                    try:
                        price = float(price_text)
                        return {
                            "product_id": product_id,
                            "competitor_price": price,
                            "source": url,
                            "timestamp": datetime.now().isoformat(),
                        }
                    except ValueError:
                        continue

        logger.warning(f"Could not extract price from {url} — falling back to mock")
        return None

    except requests.RequestException as e:
        logger.warning(f"Scraping failed for {url}: {e} — falling back to mock")
        return None


def fetch_competitor_price(product_id: str, use_scraper: bool = False) -> dict:
    """
    Fetch competitor price — tries real scraping if enabled, falls back to mock.
    """
    if use_scraper:
        for source, url_template in MOCK_URLS.items():
            url = url_template.format(product_id=product_id)
            result = scrape_price_from_url(url, product_id)
            if result:
                return result

    # Fall back to mock API (always works)
    return mock_fetch(product_id)


def fetch_all_competitor_prices(use_scraper: bool = False) -> list:
    """Fetch prices for all configured products."""
    results = []
    for pid in config.PRODUCT_IDS:
        result = fetch_competitor_price(pid, use_scraper=use_scraper)
        results.append(result)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    prices = fetch_all_competitor_prices(use_scraper=False)
    for p in prices[:5]:
        print(f"  {p['product_id']}: ${p['competitor_price']:.2f} from {p['source']}")
