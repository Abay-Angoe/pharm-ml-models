"""The active class subset: 20 classes from the NIH C3PI Training image set.

Single source of truth for class ids, human labels and look-alike groups, imported
by preprocess/dataset/evaluate/gradcam so they cannot drift apart.

Class unit is **NDC-9** (labeler + product), not the 11-digit NDC. The last two
digits of an 11-digit NDC are the *package size*, which does not change what the
tablet looks like, so NDC-11 needlessly splits one physical pill across classes.

Selection rules (spec sections 1-2):
  * source: C3PI_Test images - the consumer-grade photographs, JPG only
  * counts are **unique image paths**, not index rows. The index lists some
    photographs twice (sometimes under two NDC-11 package variants, sometimes
    twice under the same NDC-11 with different name strings). An earlier
    selection counted rows and inflated nine products by exactly 2x; those
    products dropped out once counted properly.
  * ranked by real image count, take the top 20
  * multi-phase products excluded. A blister pack of a triphasic contraceptive
    carries one NDC but contains differently-coloured tablets, so the label would
    be internally inconsistent and would corrupt the section 8 imprint analysis.

Measured: 1,212 images, 47-101 per class, mean 60.6. This does NOT clear section
2's >=100 target and no C3PI subset can - counted by NDC-9 on unique images, only
1 of 2,362 products reaches 100 and only 9 reach 60. Documented in the README
rather than hidden behind augmentation.
"""

from __future__ import annotations

# (ndc9, slug, human label, look-alike group)
CLASS_TABLE: list[tuple[str, str, str, str]] = [
    ("634590701", "amrix_30mg_cap",        "AMRIX CAP 30MG",                        "capsule"),
    ("000390223", "amaryl_4mg",            "AMARYL 4MG TABLETS",                    "glimepiride"),
    ("000937254", "glimepiride_generic",   "GLIMEPIRIDE Tablets USP",               "glimepiride"),
    ("005915544", "allopurinol_300mg",     "ALLOPURINOL 300MG TABS",                "solo"),
    ("576640104", "bethanechol_10mg",      "BETHANECHOL TAB 10MG",                  "solo"),
    ("000746114", "depakote_sprinkle",     "DEPAKOTE SPRINKLES 125 MG",             "capsule"),
    ("000935712", "glyburide_metformin_a", "GLYBURIDE and METFORMIN HCl (…712)",    "glyburide_metformin"),
    ("000935711", "glyburide_metformin_b", "GLYBURIDE and METFORMIN HCl (…711)",    "glyburide_metformin"),
    ("001725411", "fluconazole_100mg",     "FLUCONAZOLE 100MG TABLETS",             "solo"),
    ("634590402", "gabitril_2mg",          "GABITRIL 2MG TABLETS",                  "gabitril"),
    ("634590404", "gabitril_4mg",          "GABITRIL 4MG TABLETS",                  "gabitril"),
    ("634590416", "gabitril_16mg",         "GABITRIL 16MG TABLETS",                 "gabitril"),
    ("000060717", "hyzaar_50_12_5",        "HYZAAR TAB 50-12.5",                    "solo"),
    ("501110398", "hydralazine",           "HydrALAZINE HYDROCHLORIDE Tablets USP", "solo"),
    ("607930850", "levoxyl_0_025mg",       "LEVOXYL 0.025MG",                       "levoxyl"),
    ("607930852", "levoxyl_0_075mg",       "LEVOXYL 0.075MG",                       "levoxyl"),
    ("001724960", "flutamide_cap",         "FLUTAMIDE Capsules USP",                "capsule"),
    ("005913973", "glipizide_metformin",   "GLIP/METFORM 5-500MG TAB",              "solo"),
    ("005550997", "fludrocortisone",       "FLUDROCORTISONE ACETATE TABLETS USP",   "solo"),
    ("000932130", "nitrofurantoin_cap",    "NITROFURANTOIN Capsules USP",           "capsule"),
]

NDC9_TO_SLUG: dict[str, str] = {n: s for n, s, _, _ in CLASS_TABLE}
SLUG_TO_NDC9: dict[str, str] = {s: n for n, s, _, _ in CLASS_TABLE}
SLUG_TO_LABEL: dict[str, str] = {s: lbl for _, s, lbl, _ in CLASS_TABLE}
CLASSES: list[str] = sorted(SLUG_TO_NDC9)

# group -> slugs, for "did the error stay inside a look-alike family?" analysis.
# Grouping is by same-brand-different-strength or same-drug-different-brand, plus
# a capsule group: those are the cases where colour and shape genuinely coincide
# and the imprint is the distinguishing feature (spec section 7).
_GROUPS: dict[str, list[str]] = {}
for _ndc, _slug, _label, _grp in CLASS_TABLE:
    _GROUPS.setdefault(_grp, []).append(_slug)

# 'solo' means "no look-alike partner in this subset"; keep it out of the group
# map so a solo->solo error is correctly counted as cross-group.
SOLO: list[str] = _GROUPS.pop("solo", [])
CLASS_GROUPS: dict[str, list[str]] = _GROUPS
CLASS_TO_GROUP: dict[str, str] = {
    s: g for g, ss in CLASS_GROUPS.items() for s in ss
}

# C3PI image type + extension we accept. C3PI_Test is the consumer-grade
# photograph set; the other types are studio references or derived renders.
IMAGE_TYPE = "C3PI_Test"
EXTENSION = "JPG"

# measured unique-image counts, for cross-checking the download
EXPECTED_COUNTS: dict[str, int] = {
    "amrix_30mg_cap": 101, "amaryl_4mg": 99, "allopurinol_300mg": 79,
    "bethanechol_10mg": 79, "depakote_sprinkle": 69, "glimepiride_generic": 62,
    "glyburide_metformin_a": 56, "glyburide_metformin_b": 55,
    "fluconazole_100mg": 54, "gabitril_16mg": 53, "hyzaar_50_12_5": 52,
    "hydralazine": 52, "levoxyl_0_075mg": 52, "gabitril_2mg": 52,
    "gabitril_4mg": 52, "flutamide_cap": 50, "glipizide_metformin": 50,
    "fludrocortisone": 49, "levoxyl_0_025mg": 49, "nitrofurantoin_cap": 47,
}


if __name__ == "__main__":
    print(f"{len(CLASSES)} classes | {len(CLASS_GROUPS)} look-alike groups | "
          f"{len(SOLO)} solo")
    for g, ss in sorted(CLASS_GROUPS.items()):
        print(f"  {g:22s} ({len(ss)}) {', '.join(ss)}")
    print(f"  {'(solo)':22s} ({len(SOLO)}) {', '.join(sorted(SOLO))}")
    total = sum(EXPECTED_COUNTS.values())
    print(f"\nexpected {total} images, {min(EXPECTED_COUNTS.values())}-"
          f"{max(EXPECTED_COUNTS.values())} per class, "
          f"mean {total / len(CLASSES):.1f}")
    assert len(CLASSES) == len(CLASS_TABLE) == 20
    assert len(set(SLUG_TO_NDC9.values())) == 20, "duplicate NDC9"
    assert set(EXPECTED_COUNTS) == set(CLASSES), "EXPECTED_COUNTS out of sync"
    print("registry consistent")
