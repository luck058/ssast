#!/bin/bash
set -euo pipefail

# Config: enable subset download for quick testing
SUBSET_ENABLED=true     # set to false to download the full split
SUBSET_SIZE=20         # number of clips to download when SUBSET_ENABLED=true
SUBSET_RANDOM=true      # true=random sample, false=first N rows

echo "--- Installing dependencies (audioset-download) ---"
# Optional: activate venv if you have one
if [ -f "../venv/bin/activate" ]; then
  source ../venv/bin/activate
fi
pip install -q audioset-download

# --- 1. Create directories and download metadata ---
echo "--- Downloading AudioSet metadata ---"
mkdir -p audioset_metadata audioset_datafiles audioset_audioclips

# Evaluation set definition
wget -q -O audioset_metadata/eval_segments.csv \
  http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv

# Label mapping (SSAST dataloader needs this)
wget -q -O audioset_metadata/class_labels_indices.csv \
  http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

CSV_TO_USE="audioset_metadata/eval_segments.csv"

# --- Optional: build a small subset CSV ---
if [ "${SUBSET_ENABLED}" = true ]; then
  echo "--- Building a ${SUBSET_SIZE}-clip subset (${SUBSET_RANDOM:+random}) ---"
  SRC="audioset_metadata/eval_segments.csv"
  DST="audioset_metadata/eval_segments_subset.csv"
  HEADER=$(head -n 1 "$SRC")
  if [ "${SUBSET_RANDOM}" = true ] && command -v shuf >/dev/null 2>&1; then
    # Random N rows (keep header)
    { echo "$HEADER"; tail -n +2 "$SRC" | shuf -n "${SUBSET_SIZE}"; } > "$DST"
  else
    # First N rows (keep header)
    head -n $((SUBSET_SIZE + 1)) "$SRC" > "$DST"
  fi
  CSV_TO_USE="$DST"
fi

CSV_BASENAME=$(basename "$CSV_TO_USE")
CSV_STEM="${CSV_BASENAME%.*}"
AUDIO_DIR="audioset_audioclips/${CSV_STEM}"

echo "--- Using CSV: ${CSV_TO_USE} ---"
echo "--- Audio output dir: ${AUDIO_DIR} ---"

# --- 2. Download raw audio from YouTube ---
echo "--- Starting raw audio download (subset may still take time) ---"
audioset-download \
  --csv-file "$CSV_TO_USE" \
  --output-dir audioset_audioclips \
  --format flac \
  --jobs 4

echo "--- Audio download complete. ---"

# --- 3. Format data for SSAST ---
echo "--- Formatting data into SSAST JSON format ---"
python3 format_audioset_for_ssast.py \
  --google_csv "$CSV_TO_USE" \
  --audio_dir "$AUDIO_DIR" \
  --output_json "audioset_datafiles/audioset_${CSV_STEM}_data.json"

# --- 4. Prepare the label file ---
echo "--- Copying label file for SSAST ---"
cp audioset_metadata/class_labels_indices.csv audioset_datafiles/

# --- 5. Zip the final formatted data ---
echo "--- Zipping the final formatted data ---"
zip -qr "audioset_ssast_${CSV_STEM}.zip" audioset_datafiles/

echo "--- Process Complete! ---"
echo "SSAST-formatted data JSON: audioset_datafiles/audioset_${CSV_STEM}_data.json"
echo "Zip created: audioset_ssast_${CSV_STEM}.zip"