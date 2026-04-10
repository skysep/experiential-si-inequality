import os
import re
import time
import math
import pandas as pd
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options


def setup_driver(headless: bool = False, window_size: str = "1200, 900") -> webdriver.Chrome:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"--window-size={window_size}")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    # keep default user agent / do not spoof aggressively
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)
    return driver


def safe_find(driver, by, selector, timeout=10):
    """
    Wait up to `timeout` seconds for element to appear (polling).
    Returns element or raises TimeoutException.
    """
    t0 = time.time()
    while True:
        try:
            el = driver.find_element(by, selector)
            return el
        except Exception:
            if time.time() - t0 > timeout:
                raise TimeoutException(f"Element {selector} not found after {timeout}s")
            time.sleep(0.3)


def nearest_scrollable_ancestor(driver, el):
    return driver.execute_script("""
        function isScrollable(e){
          if (!e) return false;
          const cs = getComputedStyle(e);
          const oy = cs.overflowY;
          return (oy === 'auto' || oy === 'scroll') && e.scrollHeight > e.clientHeight;
        }
        let node = arguments[0];
        while (node && node !== document.body && node !== document.documentElement){
          node = node.parentElement;
          if (isScrollable(node)) return node;
        }
        return document.scrollingElement || document.documentElement || document.body;
    """, el)



def scroll_to_bottom_until_stable(driver, scrollable, pause=2, max_scrolls=100000):
    last_h = -1
    for scroll_round in range(max_scrolls):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scrollable)
        time.sleep(pause)
        h = driver.execute_script("return arguments[0].scrollHeight;", scrollable)
        if (h == last_h) & (scroll_round>15):
            break
        last_h = h
        # print('scroll_round: ', scroll_round)


def fetch_reviews_for_place(place_id: str, driver: webdriver.Chrome, max_reviews: int = 100000, max_scrolls:int = 100000, verbose: bool = False) -> List[Dict]:
    """
    Open Google Maps place by place_id and extract up to max_reviews visible reviews.
    Returns a list of dicts: {place_id, author, rating, text, date}.
    """
    url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
    driver.get(url)
    time.sleep(3.0) 


    # Click on Review button
    selectors = [
        "//button[contains(., 'Reviews')]",
        "//button[contains(@aria-label, 'Reviews')]",
        "//div[contains(text(), 'Reviews')]",
        "//span[contains(text(), 'Reviews')]"
    ]
    for sel in selectors:
        try:
            all_reviews_btn = safe_find(driver, 
                                        By.XPATH, 
                                        sel,
                                        timeout=20)
            driver.execute_script("arguments[0].click();", all_reviews_btn)
            time.sleep(1)
            break

        except TimeoutException:
            time.sleep(0.5)



    # Scrolling
    scrollable = None
    candidates = ("//div[contains(@class,'TFQHme')]"
                "| //div[contains(@class,'AyRUI')]"
                "| //div[contains(@class,'jftiEf')]",
                # "//div[contains(@class,'section-review')]",
                # " | //div[@role='article' and .//span]"
                )
    for sel in candidates:
        try:
            scrollables = driver.find_elements(By.XPATH, sel)
            break
        except Exception:
            continue
    if scrollables is None:
        scrollables = driver.find_elements(By.TAG_NAME, "body")
    scrollbox = nearest_scrollable_ancestor(driver, scrollables[-1])
    if scrollables:
        driver.execute_script("arguments[0].scrollIntoView({block:'end'});", scrollables[-1])
        scroll_to_bottom_until_stable(driver, scrollbox, pause=2, max_scrolls=max_scrolls)
    else:
        driver.execute_script("window.scrollBy(0, 400);")
    
    # time.sleep(1)



    # Scraping
    collected = []
    attempts = 0
    while len(collected) < max_reviews and attempts < 1:
        review_blocks = driver.find_elements(By.XPATH, "//div[contains(@class,'jftiEf')]")
        for rb in review_blocks:
            # print(len(collected))
            try:
                rating = None
                try:
                    star_span = rb.find_element(By.XPATH, ".//span[contains(@aria-label,'Rated') or contains(@aria-label,'star') or contains(@aria-label,'stars') or contains(@aria-label,'out of') or contains(@class,'fzvQIb')]")
                    # print(star_span)
                    rating_text = star_span.get_attribute("aria-label") or star_span.text.strip()
                    m = re.search(r"(\d(\.\d)?)", rating_text)
                    if m:
                        rating = float(m.group(1))
                        # print(rating)

                except Exception:
                    # try:
                    #     title_span = rb.find_element(By.XPATH, ".//span[@aria-hidden='false']")
                    #     m = re.search(r"(\d(\.\d)?)", title_span.text)
                    #     if m:
                    #         rating = float(m.group(1))
                    # except Exception:
                        continue
                        rating = None


                author = None
                try:
                    a = rb.find_element(By.XPATH, ".//div[contains(@class,'d4r55')]")
                    author = a.text.strip()
                except Exception:
                    # try:
                    #     author = rb.find_element(By.XPATH, ".//a[contains(@href,'/maps/contrib')]").text.strip()
                    # except Exception:
                        continue
                        # author = None

                date_text = None
                try:
                    d = rb.find_element(By.XPATH, ".//span[contains(@class,'rsqaWe') or contains(@class,'xRkPPb')]")
                    date_text = d.text.strip()
                except Exception:
                    # try:
                    #     header_spans = rb.find_elements(By.XPATH, ".//div[contains(@class,'d4r55')]/following-sibling::span")
                    #     if header_spans:
                    #         date_text = header_spans[-1].text.strip()
                    # except Exception:
                        continue
                        date_text = None

                review_text = None
                try:
                    # Google sometimes uses nested spans; try a few patterns
                    p = rb.find_element(By.XPATH, ".//span[contains(@class,'wiI7pd')]")
                    review_text = p.text.strip()
                except Exception:
                    # try:
                    #     review_text = rb.find_element(By.XPATH, ".//span[contains(@class,'review-text')]").text.strip()
                    # except Exception:
                        review_text = None


                key = (author, review_text, date_text, rating)
                # print(key)
                if any(k for k in collected if (k.get("key") == key)):
                    # print('err')
                    continue 

                collected.append({
                    "key": key,
                    "place_id": place_id,
                    "author": author,
                    "rating": rating,
                    "date": date_text,
                    "text": review_text
                })
                # print(collected)
                if len(collected) >= max_reviews:
                    break
            except StaleElementReferenceException:
                continue
            except Exception:
                continue

        # if enough, break
        if len(collected) >= max_reviews:
            break
        time.sleep(0.8)
        attempts += 1
        # print('Attempt: ', attempts)

    out = []
    for item in collected[:max_reviews]:
        d = item.copy()
        d.pop("key", None)
        out.append(d)

    return out


def collect_reviews(pid: str, driver, max_reviews: int = 100000, max_scrolls: int = 100000, verbose: bool = False):
    start_time = time.time()  # <-- start timer
    results = []
    try:
        reviews = fetch_reviews_for_place(pid, driver, max_reviews=max_reviews, max_scrolls=max_scrolls, verbose=True)
        if not reviews:
            results.append({"place_id": pid, "text": "no_reviews_or_could_not_open"})
        else:
            for r in reviews:
                results.append(r)
    except Exception as e:
        print(f"[ERROR] place {pid}: {e}")
        results.append({"place_id": pid, "error": str(e)})
    finally:
        out_df = pd.DataFrame(results)
    print(f"Iteration runtime: {time.time() - start_time:.2f} seconds") 
    
    return out_df