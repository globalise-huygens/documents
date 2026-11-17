"""Upload the deposit's files to the draft doi:10.34894/P826UY; skips files that are already there."""
import json, os, subprocess, sys, urllib.request
U = os.path.dirname(os.path.abspath(__file__)) + "/deposit_upload"
TOKEN = open(os.path.expanduser("~/.config/dataverse/token")).read().strip()
BASE, PID = "https://dataverse.nl/api/datasets/:persistentId", "doi:10.34894/P826UY"
DESCRIPTIONS = {
    "texts.zip": "Archive of data/texts/ (normalised transcriptions per inventory). Holds texts_folder.zip: unzip both, see README.md.",
    "layout.zip": "Archive of data/layout/ (page layout per inventory, from the PageXML). Holds layout_folder.zip: unzip both, see README.md.",
    "embeddings.zip": "Archive of embeddings/ (page embeddings of the evaluation inventories, kNN start scores, embedding features). Holds embeddings_folder.zip: unzip both, see README.md.",
}
req = urllib.request.Request(f"{BASE}/?persistentId={PID}", headers={"X-Dataverse-key": TOKEN})
have = {(f.get("directoryLabel", ""), f["dataFile"]["filename"]) for f in json.load(urllib.request.urlopen(req, timeout=120))["data"]["latestVersion"]["files"]}
done = failed = 0
for line in open(os.path.join(U, "upload_list.txt")):
    path, folder = line.rstrip("\n").split("|")
    name = os.path.basename(path)
    if (folder, name) in have:
        continue
    meta = {"tabIngest": "false"}
    if folder:
        meta["directoryLabel"] = folder
    if name in DESCRIPTIONS:
        meta["description"] = DESCRIPTIONS[name]
    r = subprocess.run(["curl", "-s", "--max-time", "7200", "-H", "@-", "-X", "POST", "-F", f"file=@{path}", "-F", "jsonData=" + json.dumps(meta),
                        f"{BASE}/add?persistentId={PID}"], input=f"X-Dataverse-key: {TOKEN}\n", capture_output=True, text=True)
    try:
        ok = json.loads(r.stdout).get("status") == "OK"
    except Exception:
        ok = False
    done += ok; failed += not ok
    print(("ok   " if ok else "FAIL ") + os.path.join(folder, name) + ("" if ok else "  " + r.stdout[:300]), flush=True)
print(f"uploaded {done}, failed {failed}, already there {len(have)}", flush=True)
