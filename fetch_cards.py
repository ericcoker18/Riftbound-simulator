"""
Fetches all cards from the Riftbound API and writes them to data/cards.json.
Run with: python fetch_cards.py
"""

import json
import urllib.request

API_BASE = "https://api.riftcodex.com/cards"
PAGE_SIZE = 50
OUTPUT = "data/cards.json"

RARITY_MAX_COPIES = {
    "Common":   3,
    "Uncommon": 3,
    "Rare":     3,
    "Epic":     2,
    "Showcase": 1,
}

# Champions are more powerful — limit to 2 copies regardless of rarity
CHAMPION_MAX_COPIES = 2


def fetch_page(page):
    url = f"{API_BASE}?dir=1&page={page}&size={PAGE_SIZE}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    })
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def map_card(card):
    attrs        = card.get("attributes", {})
    classification = card.get("classification", {})
    meta         = card.get("metadata", {})
    text         = card.get("text", {})

    card_type  = classification.get("type", "")
    supertype  = classification.get("supertype") or ""
    rarity     = classification.get("rarity", "Common")
    domains    = classification.get("domain", [])

    # Only keep Unit, Spell, and Gear cards
    if card_type not in ("Unit", "Spell", "Gear"):
        return None

    # Skip alternate art, signature, and overnumbered duplicates
    if meta.get("alternate_art") or meta.get("signature") or meta.get("overnumbered"):
        return None

    cost   = attrs.get("energy")
    power  = attrs.get("power")
    health = attrs.get("might")   # None for Spells and Gear

    # energy is required for all playable cards
    if cost is None:
        return None

    # Units require might; Spells/Gear have health=0
    if card_type == "Unit" and health is None:
        return None

    is_champion = supertype.lower() == "champion"
    max_copies  = CHAMPION_MAX_COPIES if is_champion else RARITY_MAX_COPIES.get(rarity, 3)

    # Combine card tags with domain for richer deckbuilding context
    tags = card.get("tags", []) + domains

    domain = domains[0] if domains else None

    return {
        "name":       card["name"],
        "cost":       int(cost),
        "rune_cost":  int(power) if power is not None else 0,
        "health":     int(health) if health is not None else 0,
        "card_type":  card_type,
        "supertype":  supertype,
        "domain":     domain,
        "weight":     1.0,
        "max_copies": max_copies,
        "tags":       tags,
        "ability":    text.get("plain", ""),
        "rarity":     rarity,
        "champion":   is_champion,
    }


def fetch_all():
    print("Fetching page 1...")
    first = fetch_page(1)
    total_pages = first["pages"]
    print(f"Total pages: {total_pages}")

    all_cards = []
    seen_names = set()

    for raw in first["items"]:
        card = map_card(raw)
        if card and card["name"] not in seen_names:
            all_cards.append(card)
            seen_names.add(card["name"])

    for page in range(2, total_pages + 1):
        print(f"Fetching page {page}/{total_pages}...")
        data = fetch_page(page)
        for raw in data["items"]:
            card = map_card(raw)
            if card and card["name"] not in seen_names:
                all_cards.append(card)
                seen_names.add(card["name"])

    return all_cards


def main():
    cards = fetch_all()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2)

    champions = sum(1 for c in cards if c["champion"])
    units     = sum(1 for c in cards if c["card_type"] == "Unit")
    spells    = sum(1 for c in cards if c["card_type"] == "Spell")
    gears     = sum(1 for c in cards if c["card_type"] == "Gear")
    print(f"\nDone. {len(cards)} unique cards written to {OUTPUT}")
    print(f"  Units     : {units} ({champions} Champions)")
    print(f"  Spells    : {spells}")
    print(f"  Gear      : {gears}")


if __name__ == "__main__":
    main()
