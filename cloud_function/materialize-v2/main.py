# main.py  –  materialize-v2
# Reads JSONL from structured/ and writes listings_master_v2.csv
import os, json, csv, io, logging
from flask import Request, jsonify
from google.cloud import storage

BUCKET_NAME       = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")
CSV_BLOB_NAME     = os.getenv("CSV_BLOB_NAME", "listings_master_v2.csv")

CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at", "source_txt",
    "price", "year", "make", "model", "mileage",
    "cylinders", "drive_type", "condition",
]

storage_client = storage.Client()

def materialize_http(request: Request):
    logging.getLogger().setLevel(logging.INFO)
    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f"{STRUCTURED_PREFIX}/"))
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]

    if not jsonl_blobs:
        return jsonify({"ok": False, "error": "no .jsonl files found"}), 200

    rows = []
    for blob in jsonl_blobs:
        try:
            text = blob.download_as_text()
            for line in text.strip().splitlines():
                record = json.loads(line)
                rows.append(record)
        except Exception as e:
            logging.error(f"Failed to read {blob.name}: {e}")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    out_blob = bucket.blob(CSV_BLOB_NAME)
    out_blob.upload_from_string(buf.getvalue(), content_type="text/csv")

    result = {
        "ok": True,
        "version": "materialize-v2",
        "rows_written": len(rows),
        "csv_blob": CSV_BLOB_NAME,
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
