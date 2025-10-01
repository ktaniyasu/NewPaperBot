from typing import Any, Dict, List


def combine(partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """単純な統合: 文字列フィールドは結合、配列はユニーク結合。
    後続で LLM での Reduce を導入予定。
    """
    if not partials:
        return {}
    keys = [
        "summary",
        "novelty",
        "methodology",
        "results",
        "future_work",
        "research_themes",
    ]
    out: Dict[str, Any] = {k: "" for k in keys}
    out["research_themes"] = []
    for p in partials:
        if not isinstance(p, dict):
            continue
        for k in keys:
            v = p.get(k)
            if v is None:
                continue
            if k == "research_themes" and isinstance(v, list):
                # ユニーク結合
                seen = set(out["research_themes"]) if isinstance(out["research_themes"], list) else set()
                for item in v:
                    if item not in seen:
                        seen.add(item)
                        out.setdefault("research_themes", [])
                        out["research_themes"].append(item)
            elif isinstance(v, str):
                out[k] = (out.get(k, "") + ("\n\n" if out.get(k) else "") + v).strip()
            else:
                out[k] = v
    return out
