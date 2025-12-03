#!/usr/bin/env python3
# match_item.py — step-by-step filtering only (brand → name → qty with interactive refinement)
#
# deps:
#   pip install orjson unidecode rapidfuzz numpy

import sys, re, argparse, orjson
from unidecode import unidecode
from rapidfuzz import fuzz

# ---------------- text utils ----------------
def canon(s: str) -> str:
    if not s:
        return ""
    s = unidecode(s).lower()
    s = re.sub(r"[^a-z0-9\.\-\+\sx×]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokenize(s: str):
    return [t for t in canon(s).split() if t]

def contains_any(hay: str, needles):
    H = " " + canon(hay) + " "
    return any((" " + n + " ") in H for n in needles)

# ---------------- brand similarity & gating ----------------
BRAND_STOPWORDS = {
    "the","and","&","co","company","inc","llc","ltd","corp","corporation","brands","brand",
    "foods","food","market","farms","farm","dairy","beverage","kitchen","kitchens","group",
    "organic","natural","naturals","quality","select","selects","choice","choices"
}

def _split_brand_variants(brand: str):
    """Split comma/ampersand/slash separated brand strings into parts."""
    if not brand:
        return []
    parts = re.split(r"[,&/]+", brand)
    out = []
    for p in parts:
        c = canon(p).strip()
        if c:
            out.append(c)
    return out

def brand_strong_tokens(s: str):
    return {t for t in tokenize(s) if t not in BRAND_STOPWORDS}

def brand_similarity(user_brand: str, candidate_brand: str) -> float:
    """Max token-set fuzzy similarity between user brand and any candidate brand variant."""
    if not user_brand:
        return 1.0
    ub = canon(user_brand)
    if not ub:
        return 1.0
    cand_parts = _split_brand_variants(candidate_brand) or [canon(candidate_brand)]
    sims = [fuzz.token_set_ratio(ub, part)/100.0 for part in cand_parts if part]
    return max(sims) if sims else 0.0

def brand_ok(row, user_brand: str, min_sim: float, partial_floor: float = 0.50) -> bool:
    # Strict brand gate (used only when no exact-variant match exists in the catalog)
    if not user_brand.strip():
        return True
    cand_brand = row.get("brand", "")
    sim = brand_similarity(user_brand, cand_brand)
    if sim >= float(min_sim):
        return True
    u_tok = brand_strong_tokens(user_brand)
    c_tok = brand_strong_tokens(cand_brand)
    strong_overlap = len(u_tok & c_tok) >= 1
    return strong_overlap and sim >= float(partial_floor)

def has_exact_brand_variant(row, user_brand: str) -> bool:
    """True if ANY variant in row['brand'] exactly equals the user brand (canonical)."""
    if not user_brand.strip():
        return False
    target = canon(user_brand)
    raw = row.get("brand", "")
    parts = _split_brand_variants(raw) or [canon(raw)]
    for part in parts:
        if part and part == target:
            return True
    return False

# ---------------- quantity parsing & matching ----------------

# Fraction normalization: "1/2" -> "0.5", "3/4" -> "0.75"
_FRACTION_RE = re.compile(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)')

def _norm_qty_string(q: str) -> str:
    """Normalize quantity string but KEEP enough structure for numeric parsing."""
    if not q:
        return ""
    s = unidecode(q).lower()

    # convert fractions like "1/2" to decimal
    def _repl(m):
        num = float(m.group(1))
        den = float(m.group(2))
        if den == 0:
            return "0"
        return f"{num/den:g}"

    s = _FRACTION_RE.sub(_repl, s)
    # normalize separators
    s = s.replace(",", " ")
    return s

# number + unit; unit can contain letters, dots, hyphens, spaces, parentheses
_NUM_UNIT = re.compile(r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z][a-zA-Z\.\-\s\(\)]*)')

def _collapse_unit(u: str) -> str:
    return re.sub(r'[^a-z]', '', (u or '').lower())

UNIT_ALIAS = {
    "floz":"fluid_ounce", "flounce":"fluid_ounce",
    "oz":"ounce",
    "ml":"milliliter", "millilitre":"milliliter", "cl":"milliliter",
    "l":"liter", "lt":"liter",
    "g":"gram", "gr":"gram",
    "kg":"kilogram", "mg":"milligram",
    "lb":"pound", "lbs":"pound",
    "ct":"count", "pack":"count", "pk":"count"
}
TO_BASE = {
    ("ounce","gram"):28.3495, ("pound","gram"):453.592, ("kilogram","gram"):1000.0, ("milligram","gram"):1/1000.0,
    ("fluid_ounce","milliliter"):29.5735, ("liter","milliliter"):1000.0
}
BASE_OF = {
    "ounce":"gram","pound":"gram","kilogram":"gram","gram":"gram","milligram":"gram",
    "fluid_ounce":"milliliter","liter":"milliliter","milliliter":"milliliter",
    "count":"count"
}

def _to_base(val, canon_unit):
    base = BASE_OF.get(canon_unit, canon_unit)
    if canon_unit != base:
        fac = TO_BASE.get((canon_unit, base))
        if fac:
            val *= fac
    return val, base

def parse_qty(q: str):
    """
    Parse a quantity string into approximate mass (grams), volume (milliliters),
    and/or count. Handles strings like '1/2 gallon (1.89 L)', '24 336g', '12 ct', etc.
    """
    if not q:
        return {"mass": None, "volume": None, "count": None}

    s = _norm_qty_string(q)
    # run a second pass through canon to normalize spaces / strip junk,
    # but after we've already converted fractions
    s = canon(s)

    mass, vol, cnt = None, None, None
    pack = None

    # pack-like prefixes: "24x", "24 ct", etc.
    pm = re.search(r'(\d+)\s*(?:x|×|pk|pack|ct)\b', s)
    if pm:
        pack = int(pm.group(1))
    # patterns like "24 336g" where 24 is a leading count
    leading_ct = re.match(r'^\s*(\d{1,3})\s+[a-zA-Z]*\s*\d', s)
    if (not pack) and leading_ct:
        pack = int(leading_ct.group(1))

    for m in _NUM_UNIT.finditer(s):
        val = float(m.group(1))
        unit_raw = _collapse_unit(m.group(2))
        if unit_raw == "cl":
            val *= 10.0  # cl -> ml
        canon_u = UNIT_ALIAS.get(unit_raw, unit_raw)
        base_val, base = _to_base(val, canon_u)
        mult = pack if pack else 1
        if base == "gram":
            mass = max(mass or 0, base_val * mult)
        elif base == "milliliter":
            vol = max(vol or 0, base_val * mult)
        elif base == "count":
            cnt = max(cnt or 0, int(val))

    if cnt is None and pack is not None:
        cnt = pack

    return {"mass": mass, "volume": vol, "count": cnt}

def qty_close(user_q: str, cat_q: str, tol=0.18):
    """
    Coarse filter: accept if ANY comparable dimension is within ±tol relative error.
    Neutral if catalog size is missing (do NOT reject for missing).
    Used for soft filtering, not for strict equality.
    """
    if not user_q.strip():
        return True
    U, C = parse_qty(user_q), parse_qty(cat_q)
    compared = False
    for k in ("mass","volume","count"):
        u, c = U[k], C[k]
        if u is None or c is None:
            continue
        compared = True
        rel = abs(u - c) / max(c, u, 1e-6)
        if rel <= tol:
            return True
    return not compared

def qty_numeric_similarity(user_q: str, cat_q: str):
    """
    Strict numeric similarity in [0,1] between user quantity and catalog quantity.
    1.0 means numerically equal (up to floating precision).
    None means no comparable numeric dimension.
    """
    U, C = parse_qty(user_q), parse_qty(cat_q)
    sims = []

    def _sim(u, c):
        if u is None or c is None:
            return None
        rel = abs(u - c) / max(u, c, 1e-6)
        return max(0.0, 1.0 - rel)

    for k in ("mass","volume","count"):
        s = _sim(U[k], C[k])
        if s is not None:
            sims.append(s)
    return max(sims) if sims else None

# ---------------- catalog I/O ----------------
def load_catalog(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            d = orjson.loads(line)
            rows.append({
                "code": d.get("code",""),
                "name": d.get("product_name",""),
                "brand": d.get("brands",""),
                "qty": d.get("quantity",""),
                "keywords": " ".join(d.get("keywords",[]) or [])
            })
    return rows

# ---------------- filtering steps ----------------
def filter_brand(rows, brand, strict_min=0.60, partial_floor=0.50):
    """
    BRAND FILTER with exact-match priority:
      1) If ANY rows contain a brand VARIANT that equals the user brand (canonical, case/diacritics-insensitive),
         keep ONLY those rows.
      2) Else, keep rows that pass the strict brand gate (brand_ok).
      3) If still empty, the caller should request clarification and show suggestions.
    """
    if not brand.strip():
        return rows

    # Step 1: exact brand-variant match → hard keep-only
    exact = [r for r in rows if has_exact_brand_variant(r, brand)]
    if exact:
        return exact

    # Step 2: strict gate (fuzzy + strong-token overlap)
    kept = [r for r in rows if brand_ok(r, brand, strict_min, partial_floor)]
    return kept

def best_brand_suggestions(rows, brand, k=5, suggest_min=0.45):
    scored = []
    for i, r in enumerate(rows):
        sim = brand_similarity(brand, r.get("brand",""))
        if sim >= suggest_min:
            scored.append((sim, i))
    if not scored:
        scored = [(brand_similarity(brand, r.get("brand","")), i) for i, r in enumerate(rows)]
    scored.sort(reverse=True)
    return [rows[i] for s,i in scored[:k]]

# Name filter: require a hit on PRODUCT NAME (keywords cannot be the sole reason)
def name_ok(row, user_name: str, name_min: float = 0.62) -> bool:
    if not user_name.strip():
        return True
    ut = set(tokenize(user_name))
    name_tokens = set(tokenize(row.get("name","")))
    if ut & name_tokens:
        return True
    sim = fuzz.token_set_ratio(canon(user_name), canon(row.get("name","")))/100.0
    return sim >= name_min

def filter_name(rows, name):
    if not name.strip():
        return rows
    kept = [r for r in rows if name_ok(r, name)]
    return kept

def best_name_suggestions(rows, name, k=5):
    scored = [(fuzz.token_set_ratio(canon(name), canon(r.get("name","")))/100.0, i)
              for i,r in enumerate(rows)]
    scored.sort(reverse=True)
    return [rows[i] for s,i in scored[:k]]

def print_block(title, rows):
    print(f"\n[{title}]")
    if not rows:
        print("  (none)")
        return
    for i,r in enumerate(rows,1):
        print(f" {i:>2}. {r['brand']} — {r['name']} ({r['qty']}) [code={r['code']}]")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Filtering-only pipeline: brand → name → qty")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--brand", default="")
    ap.add_argument("--product", default="")
    ap.add_argument("--quantity", default="")
    ap.add_argument("--max_show", type=int, default=5)
    # brand strictness knobs
    ap.add_argument("--brand_min", type=float, default=0.60,
                    help="strict brand similarity threshold to keep (used only when no exact brand match exists)")
    ap.add_argument("--brand_partial_floor", type=float, default=0.50,
                    help="lower fuzzy floor when strong-token overlap exists (used only when no exact brand match)")
    ap.add_argument("--brand_suggest_min", type=float, default=0.45,
                    help="min similarity to show as suggestion when no strict matches")
    args = ap.parse_args()

    rows = load_catalog(args.catalog)

    # 1) BRAND (exact-variant priority; else strict gate; else suggestions)
    working = filter_brand(rows, args.brand,
                           strict_min=args.brand_min,
                           partial_floor=args.brand_partial_floor)
    if not working:
        sugg = best_brand_suggestions(rows, args.brand,
                                      k=args.max_show,
                                      suggest_min=args.brand_suggest_min)
        print_block("brand suggestions (no strict match)", sugg)
        print("\n[status] need brand clarification")
        sys.exit(0)
    if 1 < len(working) <= args.max_show:
        print_block("shortlist after brand", working)

    # 2) NAME
    working2 = filter_name(working, args.product)
    if not working2:
        sugg = best_name_suggestions(working, args.product, k=args.max_show)
        print_block("name suggestions (no strict name match)", sugg)
        print("\n[status] need product-name clarification")
        sys.exit(0)
    working = working2
    if 1 < len(working) <= args.max_show:
        print_block("shortlist after name", working)

    # 3) OPTIONAL CLI QUANTITY (soft filter only; coarse narrowing)
    if args.quantity.strip():
        soft = [r for r in working if qty_close(args.quantity, r.get("qty",""))]
        if soft:
            working = soft

    # If too many remain → interactive quantity refinement
    if len(working) > args.max_show:
        print_block("Options", working[:args.max_show])
        qty_input = input(
            "\nMore than one item found. Enter quantity to refine "
            "(e.g., '1.85 l', '1/2 gallon', '12 ct'), or press ENTER to select manually: "
        ).strip()

        if qty_input:
            # compute numeric similarity for each candidate
            sims = [qty_numeric_similarity(qty_input, r.get("qty","")) for r in working]
            pairs = [(s,i) for i,s in enumerate(sims) if s is not None]

            if pairs:
                max_sim = max(s for s,i in pairs)
                # STRICT group: nearly best AND almost exact numeric match
                STRICT_THRESH = 0.99
                strict_idx = [i for s,i in pairs if (s >= max_sim - 1e-6) and (s >= STRICT_THRESH)]

                if len(strict_idx) == 1:
                    # single clear numeric match → auto-select
                    working = [working[strict_idx[0]]]
                    print_block("FINAL", working)
                    print("\n[status] single match")
                    sys.exit(0)
                elif len(strict_idx) > 1:
                    # multiple items share the same numeric size (e.g., several 1.89 L milks)
                    narrowed = [working[i] for i in strict_idx]
                    working = narrowed
                    print("\nQuantity narrowed candidates, but multiple remain.")
                    print_block("Options", working[:args.max_show])

                    choice = input("Choose item # (or press ENTER to cancel and keep all): ").strip()
                    if choice:
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(working):
                                working = [working[idx]]
                        except Exception:
                            print("Please enter a valid number.")
                else:
                    # we had numeric info but nothing close enough to be strict; fall through
                    pass

        # If STILL too many candidates → simple manual choice
        if len(working) > args.max_show:
            print("\nStill multiple candidates; please choose an item:")
            print_block("Options", working[:args.max_show])
            choice = input("Choose item # (or press ENTER to keep all): ").strip()
            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(working):
                        working = [working[idx]]
                except Exception:
                    print("Please enter a valid number.")

    # If 1 < len <= max_show and no quantity refinement happened (or not needed)
    elif 1 < len(working) <= args.max_show and not args.quantity.strip():
        print_block("Options", working)
        choice = input("\nMore than one item found. Choose item # (or press ENTER to keep all): ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(working):
                    working = [working[idx]]
            except Exception:
                print("Please enter a valid number.")

    # Final
    print_block("FINAL", working[:args.max_show])
    if len(working) == 1:
        print("\n[status] single match")
        sys.exit(0)
    elif len(working) == 0:
        print("\n[status] no candidates")
        sys.exit(10)
    else:
        print("\n[status] multiple candidates")
        sys.exit(0)

if __name__ == "__main__":
    main()