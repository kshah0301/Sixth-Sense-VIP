#!/usr/bin/env python3
import argparse
import json
import sys

from match_item_ import (
    build_brand_token_index,
    download_image,
    load_catalog,
    match_freeform_item,
    off_search_top_products,
    print_block,
)


def _load_ingredients_json(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("ingredients"), list):
        return [str(x) for x in data["ingredients"]]
    raise ValueError("Expected a JSON list or an object with an 'ingredients' list.")


def main():
    ap = argparse.ArgumentParser(
        description="Match recipe ingredients to OpenFoodFacts catalog; optionally download images."
    )
    ap.add_argument("--catalog", default="off_us.jsonl", help="Path to OpenFoodFacts jsonl catalog")
    ap.add_argument("--meal", default="", help="Meal description (uses recipes.get_recipe_ingredients)")
    ap.add_argument("--ingredients_json", default="", help="JSON file with ingredient list")
    ap.add_argument("--download_images", action="store_true", help="Download images for matched items")
    ap.add_argument("--images_dir", default="images", help="Output directory for downloaded images")
    ap.add_argument("--max_show", type=int, default=5, help="How many candidates to show on ambiguity")
    ap.add_argument(
        "--off_choose_top5",
        action="store_true",
        help="Instead of local catalog matching, query OpenFoodFacts and let you choose 1 of the top 5 results per ingredient.",
    )
    args = ap.parse_args()

    if not args.meal and not args.ingredients_json:
        print("Provide either --meal or --ingredients_json.", file=sys.stderr)
        return 2
    if args.meal and args.ingredients_json:
        print("Provide only one of --meal or --ingredients_json.", file=sys.stderr)
        return 2

    if args.meal:
        from recipes import get_recipe_ingredients

        ingredients = get_recipe_ingredients(args.meal)
    else:
        ingredients = _load_ingredients_json(args.ingredients_json)

    if not ingredients:
        print("No ingredients found.", file=sys.stderr)
        return 2

    results = []
    if args.off_choose_top5:
        print(f"[info] ingredients={len(ingredients)} mode=off_choose_top5")
        for item in ingredients:
            cands = off_search_top_products(item, limit=5)
            if not cands:
                print(f"\n[NO OFF RESULTS] {item}")
                results.append(
                    {
                        "status": "none",
                        "query": item,
                        "brand_guess": "",
                        "name_guess": item,
                        "qty_guess": "",
                        "best": None,
                        "candidates": [],
                    }
                )
                continue

            print(f"\n[OFF top 5] {item}")
            print_block("candidates", cands)
            choice = input("Choose item # (1-5, ENTER=1): ").strip()
            try:
                idx = int(choice) - 1 if choice else 0
            except Exception:
                idx = 0
            idx = max(0, min(len(cands) - 1, idx))
            best = cands[idx]

            res = {
                "status": "single",
                "query": item,
                "brand_guess": "",
                "name_guess": item,
                "qty_guess": "",
                "best": best,
                "candidates": cands,
            }
            results.append(res)

            print(
                f"[CHOSEN] {best.get('brand','')} — {best.get('name','')} ({best.get('qty','')}) [code={best.get('code','')}]"
            )
            if args.download_images:
                download_image(best, out_dir=args.images_dir)
    else:
        rows = load_catalog(args.catalog)
        brand_token_index = build_brand_token_index(rows)
        print(f"[info] ingredients={len(ingredients)} catalog_rows={len(rows)} mode=local_match")
        for item in ingredients:
            res = match_freeform_item(rows, item, brand_token_index=brand_token_index, max_candidates=30)
            results.append(res)
            best = res["best"]
            if not best:
                print(f"\n[NO MATCH] {item}")
                continue

            print(
                f"\n[MATCH:{res['status']}] {item}\n"
                f"  => {best.get('brand','')} — {best.get('name','')} ({best.get('qty','')}) [code={best.get('code','')}]"
            )

            if res["status"] != "single":
                print_block("candidates", res["candidates"][: args.max_show])

            if args.download_images:
                download_image(best, out_dir=args.images_dir)

    # machine-friendly output if caller wants it
    print("\n[json]")
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
