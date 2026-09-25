"""
Export the frozen analysis snapshot used by the manuscript
==========================================================
Dumps every warehouse input the P4 analyses read into p4_early_warning/frozen/,
so the paper can be reproduced with MPOX_DATA=frozen and no database access.

Contents (derived, aggregate, openly sourced data only — no post text):
  features_weekly.csv                 state-week feature store (all rows)
  surveillance_weekly.csv             ingested NCDC state-week reports
  social_weekly_counts.csv            mpox-relevant posts per ISO week (counts only),
                                      pinned to data_access.SOCIAL_SNAPSHOT_CUTOFF
  social_languages.csv                posts per detected language (same pin)
  gbif_rodentia_effort_by_state.json  GBIF Rodentia records per state (effort proxy)
  MANIFEST.json                       row counts + SHA-256 of each file

Run:  MPOX_DATA=db python p4_early_warning/export_frozen_snapshot.py
"""
import os, sys, json, hashlib, shutil
from datetime import datetime, timezone

os.environ["MPOX_DATA"] = "db"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_access as da


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(da.FROZEN_DIR, exist_ok=True)
    out = {
        "features_weekly.csv": da._all_feature_rows(),
        "surveillance_weekly.csv": da.load_surveillance(),
        "social_weekly_counts.csv": da.load_social_weekly(),
        "social_languages.csv": da.load_social_languages(),
    }
    for name, df in out.items():
        df.to_csv(os.path.join(da.FROZEN_DIR, name), index=False, lineterminator="\n")
        print(f"  {name:32s} {len(df):>6} rows")

    # GBIF effort: reuse the fetch cache written by reservoir_robustness.py
    legacy_cache = os.path.join(da.HERE, "models", "_rodentia_effort_cache.json")
    if not os.path.exists(da.EFFORT_FILE) and os.path.exists(legacy_cache):
        shutil.copyfile(legacy_cache, da.EFFORT_FILE)

    files = sorted(f for f in os.listdir(da.FROZEN_DIR) if f != "MANIFEST.json")
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "social_snapshot_cutoff": da.SOCIAL_SNAPSHOT_CUTOFF,
        "files": {f: {"sha256": sha256(os.path.join(da.FROZEN_DIR, f)),
                      "rows": int(len(out[f])) if f in out else None} for f in files},
    }
    with open(os.path.join(da.FROZEN_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved -> {da.FROZEN_DIR}")


if __name__ == "__main__":
    main()
