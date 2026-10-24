"""轻量文本工具：编辑距离、正则槽位匹配。"""

import re
from typing import Iterable, Optional


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def normalized_similarity(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def _longest_substring_hit(text: str, candidates: Iterable[str]) -> Optional[str]:
    text = text.strip().lower()
    hits = [c for c in candidates if str(c).strip().lower() in text]
    if not hits:
        return None
    return max(hits, key=lambda c: len(str(c)))


def best_match_in_candidates(
    text: str, candidates: Iterable[str], threshold: float = 0.55
) -> Optional[str]:
    text = text.strip().lower()
    hit = _longest_substring_hit(text, candidates)
    if hit:
        return hit
    best_val, best_score = None, -1.0
    for cand in candidates:
        cand_l = str(cand).strip().lower()
        score = normalized_similarity(text, cand_l)
        if score > best_score:
            best_score, best_val = score, cand
    if best_score >= threshold:
        return best_val
    return None


def best_substring_match(text: str, candidates: Iterable[str], threshold: float = 0.6) -> Optional[str]:
    """在用户话术中寻找与候选值最相近的子串（用于枚举槽）。"""
    text = text.strip().lower()
    hit = _longest_substring_hit(text, candidates)
    if hit:
        return hit
    best_val, best_score = None, -1.0
    for cand in candidates:
        cand_l = str(cand).strip().lower()
        for i in range(len(text)):
            for j in range(i + 1, len(text) + 1):
                span = text[i:j]
                if len(span) < max(1, len(cand_l) // 2):
                    continue
                score = normalized_similarity(span, cand_l)
                if score > best_score:
                    best_score, best_val = score, cand
    if best_score >= threshold:
        return best_val
    return None


def regex_slot_value(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text)
    if m:
        return m.group(0)
    if "0-9" in pattern:
        m = re.search(r"\d{1,2}", text)
        if m:
            return m.group(0)
    return None
