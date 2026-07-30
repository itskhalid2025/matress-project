"""
reconcile.py — 3-way identity reconciliation: texture vs banner vs QR.

Policy (strict): PASS only when all three sources are present AND agree.
Anything else is a non-pass, categorized by what went wrong so the
dashboard/logs can tell the story.

Sources:
  texture — fabric classifier output (ground truth candidate, but still a claim here)
  banner  — OCR of printed sash on top surface (cam2)
  qr      — decoded QR payload from side label (cam0)

No third-party deps. Pure stdlib.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re

# ---------------------------------------------------------------------------
# Canonical SKU normalization
# ---------------------------------------------------------------------------

CANONICAL_SKUS = {"gravite", "maxi_plush", "ortholex", "maxi_pro",
                  "purity_plus", "memorise"}  # last two parked, but map them anyway

# Alias table: lowercase, alphanumeric-only keys -> canonical SKU.
# Covers banner text, QR productName values, and label variety strings.
# NOTE: "maxiplushpro" MUST resolve to maxi_pro, not maxi_plush —
# the physical banner reads "MAXIPLUSH PRO".
_ALIASES = {
    # gravite
    "gravite": "gravite",
    "gravitemattress": "gravite",
    # maxi_plush
    "maxiplush": "maxi_plush",
    "maxiplushmattress": "maxi_plush",
    # ortholex
    "ortholex": "ortholex",
    "ortholexmattress": "ortholex",
    # maxi_pro  (banner says "Maxiplush Pro"; label variety says "MAXI PRO")
    "maxipro": "maxi_pro",
    "maxiplushpro": "maxi_pro",
    "maxiplushpromattress": "maxi_pro",
    # parked SKUs
    "purityplus": "purity_plus",
    "memorise": "memorise",
    "memorisemattress": "memorise",
}


def _squash(s: str) -> str:
    """Lowercase and strip everything non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Every alias key must already be in squashed form, else exact-match lookup
# silently fails. Assert at import time so a typo is caught immediately.
for _k in _ALIASES:
    assert _k == _squash(_k), f"alias key not squash-normalized: {_k!r}"

# Master lookup: squashed canonical SKUs + aliases. Building it from the
# canonical set (rather than checking CANONICAL_SKUS directly) means a
# canonical id always resolves even if its alias row is ever missing.
_LOOKUP = {_squash(sku): sku for sku in CANONICAL_SKUS}
_LOOKUP.update(_ALIASES)

# Precompute the substring-scan order once (longest first) instead of
# re-sorting on every call.
_LOOKUP_BY_LEN = sorted(_LOOKUP, key=len, reverse=True)


def normalize_sku(raw: Optional[str]) -> Optional[str]:
    """
    Map a raw string (banner OCR text, QR productName, classifier label)
    to a canonical SKU id. Returns None if unrecognized or empty.

    Longest key first on the substring pass so 'maxiplushpro' wins over
    'maxiplush' when both are contained in the input — this is the property
    that keeps maxi_pro from being misread as maxi_plush.
    """
    if not raw:
        return None
    sq = _squash(raw)
    if not sq:
        return None
    if sq in _LOOKUP:                   # exact hit (canonical or alias)
        return _LOOKUP[sq]
    # substring fallback for noisy OCR ("mmfoam maxiplush pro", QR with
    # "_-_South" suffix, etc.)
    for key in _LOOKUP_BY_LEN:
        if key in sq:
            return _LOOKUP[key]
    return None


def sku_from_qr_payload(payload: Optional[str]) -> Optional[str]:
    """
    Extract SKU from QR payload of the form:
      productName=Gravite_Mattress&batchNo=...&inventoryItemId=...
    Tolerant of missing keys / malformed payloads -> None.
    """
    if not payload:
        return None
    m = re.search(r"productName=([^&]+)", payload, re.IGNORECASE)
    if not m:
        # fall back: try normalizing the whole payload
        return normalize_sku(payload)
    return normalize_sku(m.group(1))


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class Verdict(Enum):
    PASS = "PASS"                        # all three present and identical
    MISMATCH = "MISMATCH"                # one claim disagrees with texture
    DOUBLE_MISMATCH = "DOUBLE_MISMATCH"  # banner & QR agree with each other, both != texture
    CONFLICT = "CONFLICT"                # all three present, all three different
    INCOMPLETE = "INCOMPLETE"            # >=1 source missing (strict policy: cannot PASS)
    NO_TEXTURE = "NO_TEXTURE"            # texture itself failed/rejected — nothing to verify against


# Verdicts that should fire the buzzer / require operator attention.
ALERT_VERDICTS = {Verdict.MISMATCH, Verdict.DOUBLE_MISMATCH,
                  Verdict.CONFLICT, Verdict.NO_TEXTURE}


@dataclass
class Reconciliation:
    verdict: Verdict
    texture: Optional[str]
    banner: Optional[str]
    qr: Optional[str]
    missing: list        # which sources were absent, e.g. ["qr"]
    disagreeing: list    # which claim sources disagree with texture
    detail: str          # human-readable one-liner for logs/dashboard

    @property
    def alert(self) -> bool:
        return self.verdict in ALERT_VERDICTS

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "alert": self.alert,
            "texture": self.texture,
            "banner": self.banner,
            "qr": self.qr,
            "missing": self.missing,
            "disagreeing": self.disagreeing,
            "detail": self.detail,
        }


def reconcile(texture_raw: Optional[str],
              banner_raw: Optional[str],
              qr_payload: Optional[str]) -> Reconciliation:
    """
    Run the strict 3-way check for one mattress.

    Inputs are RAW values straight from each subsystem:
      texture_raw : classifier output (canonical id, or None if rejected)
      banner_raw  : OCR text from banner region, or None if not detected
      qr_payload  : raw decoded QR string, or None if no decode

    Strict policy: PASS requires texture == banner == qr, all present.
    """
    t = normalize_sku(texture_raw)
    b = normalize_sku(banner_raw)
    q = sku_from_qr_payload(qr_payload)

    missing = [name for name, val in (("banner", b), ("qr", q)) if val is None]

    # Texture failed entirely — nothing to anchor on.
    if t is None:
        return Reconciliation(
            Verdict.NO_TEXTURE, t, b, q,
            missing=["texture"] + missing, disagreeing=[],
            detail="Texture classification failed/rejected; manual check required.")

    disagreeing = [name for name, val in (("banner", b), ("qr", q))
                   if val is not None and val != t]

    # --- All three present ---
    if not missing:
        if not disagreeing:
            return Reconciliation(
                Verdict.PASS, t, b, q, [], [],
                detail=f"All three sources agree: {t}.")
        if len(disagreeing) == 2:
            if b == q:
                return Reconciliation(
                    Verdict.DOUBLE_MISMATCH, t, b, q, [], disagreeing,
                    detail=(f"Banner and QR both claim '{b}' but texture says "
                            f"'{t}'. Two independent claims vs fabric — "
                            f"possible wrong cover or classifier error. STOP LINE."))
            return Reconciliation(
                Verdict.CONFLICT, t, b, q, [], disagreeing,
                detail=(f"Three-way disagreement: texture={t}, banner={b}, "
                        f"qr={q}. Manual check required."))
        # exactly one disagrees
        bad = disagreeing[0]
        bad_val = b if bad == "banner" else q
        return Reconciliation(
            Verdict.MISMATCH, t, b, q, [], disagreeing,
            detail=f"{bad} claims '{bad_val}' but texture (and the other "
                   f"source) say '{t}'.")

    # --- One or both claims missing: strict policy forbids PASS ---
    if disagreeing:
        # Even with a source missing, an active disagreement outranks
        # incompleteness — surface it as a mismatch.
        bad = disagreeing[0]
        bad_val = b if bad == "banner" else q
        return Reconciliation(
            Verdict.MISMATCH, t, b, q, missing, disagreeing,
            detail=f"{bad} claims '{bad_val}' vs texture '{t}' "
                   f"(also missing: {', '.join(missing)}).")

    return Reconciliation(
        Verdict.INCOMPLETE, t, b, q, missing, [],
        detail=f"Texture={t}; missing {', '.join(missing)} — cannot PASS "
               f"under strict all-three policy. Manual verify.")


# ---------------------------------------------------------------------------
# Session tally for the POC dashboard / end-of-run summary
# ---------------------------------------------------------------------------

class Tally:
    def __init__(self):
        self.records = []

    def add(self, rec: Reconciliation):
        self.records.append(rec)

    def summary(self) -> dict:
        out = {v.value: 0 for v in Verdict}
        for r in self.records:
            out[r.verdict.value] += 1
        out["total"] = len(self.records)
        out["alerts"] = sum(1 for r in self.records if r.alert)
        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # QR payloads use the REAL confirmed decodes:
    #   gravite    -> Gravite_Mattress
    #   ortholex   -> Ortho_Lex_Mattress
    #   maxi_plush -> Maxi_Plush_Mattress_-_South
    #   maxi_pro   -> Maxipro_Mattress_-_South
    # Tested both bare and with a productName= wrapper since the exact
    # payload envelope isn't confirmed here.
    cases = [
        # (texture, banner_ocr, qr_payload, expected verdict)
        ("gravite", "GRAVITE", "Gravite_Mattress", Verdict.PASS),
        ("gravite", "MM FOAM GRAVITE", "productName=Gravite_Mattress&batchNo=1", Verdict.PASS),
        ("ortholex", "ORTHOLEX", "Ortho_Lex_Mattress", Verdict.PASS),
        ("maxi_pro", "MAXIPLUSH PRO", "Maxipro_Mattress_-_South", Verdict.PASS),
        ("maxi_plush", "MAXIPLUSH", "Maxi_Plush_Mattress_-_South", Verdict.PASS),
        # 0% QR-decode SKUs: missing QR can never PASS under strict policy
        ("maxi_plush", "MAXIPLUSH", None, Verdict.INCOMPLETE),
        ("maxi_pro", "MAXIPLUSH PRO", None, Verdict.INCOMPLETE),
        ("ortholex", None, "Ortho_Lex_Mattress", Verdict.INCOMPLETE),
        # genuine mismatches
        ("ortholex", "MAXIPLUSH PRO", "Ortho_Lex_Mattress", Verdict.MISMATCH),
        ("gravite", "ORTHOLEX", "Ortho_Lex_Mattress", Verdict.DOUBLE_MISMATCH),
        ("gravite", "ORTHOLEX", "Maxi_Plush_Mattress_-_South", Verdict.CONFLICT),
        ("gravite", "ORTHOLEX", None, Verdict.MISMATCH),        # disagreement outranks missing
        (None, "GRAVITE", "Gravite_Mattress", Verdict.NO_TEXTURE),
        # KNOWN RISK: OCR drops the small green "PRO" on a maxi_pro banner,
        # so banner normalizes to maxi_plush and disagrees with texture+QR.
        # Produces a MISMATCH (false alarm) — the safe failure direction.
        ("maxi_pro", "MAXIPLUSH", "Maxipro_Mattress_-_South", Verdict.MISMATCH),
        # noisy OCR still normalizes
        ("gravite", "mm foam GRAVITE wake up positive", None, Verdict.INCOMPLETE),
    ]
    tally = Tally()
    ok = True
    for t_, b_, q_, exp in cases:
        r = reconcile(t_, b_, q_)
        tally.add(r)
        status = "ok " if r.verdict == exp else "FAIL"
        if r.verdict != exp:
            ok = False
        print(f"[{status}] {r.verdict.value:<15} <- t={t_!r} b={b_!r} q={q_!r}")
        print(f"       {r.detail}")
    print("\nSummary:", tally.summary())
    raise SystemExit(0 if ok else 1)