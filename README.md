# Video-traffic-agent

Classify streaming-platform video traffic (Netflix / Stan / YouTube) using
vector-similarity search over statistical features, powered by Zilliz (Milvus)
and an OpenAI GPT-4o agent.

---

## Project layout

```
video_traffic_agent/
├── dataset/
│   ├── train/          # 3 platforms × 20 videos × 80 samples
│   └── test/           # 3 platforms × 20 videos × 20 samples
├── utils/
│   └── stats_utils.py  # shared timeseries & stats utilities
├── ingest.py           # Script 1 – build DB from training data
├── agent.py            # Script 2 – OpenAI agent + folder watcher
├── requirements.txt
├── .env.example
└── watch_folder/       # drop test CSVs here (created automatically)
```

---

## Prerequisites

### 1. Python virtual environment

Create and activate a virtual environment before installing any packages:

```bash
# Create the venv (run once from the project root)
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Verify
python --version                 # should print Python 3.10+
```

> Keep the venv activated for all subsequent steps. To deactivate later run `deactivate`.

---

### 2. OpenAI API key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Click **Create new secret key**, give it a name, and copy the key.
3. Paste it into your `.env` file as `OPENAI_API_KEY=sk-...`.

---

### 3. Zilliz Cloud cluster

Zilliz Cloud is the managed cloud service for Milvus. A free tier is available.

1. **Sign up** at [https://cloud.zilliz.com](https://cloud.zilliz.com) (Google / GitHub login supported).
2. **Create a project** — click **+ New Project**, give it a name (e.g. `video_traffic`), and click **Create**.
3. **Create a cluster** inside the project:
   - Click **+ Create Cluster**.
   - Choose the **Free** tier (Serverless) — sufficient for this project.
   - Select a cloud region close to you.
   - Click **Create** and wait ~1–2 minutes for the cluster to become **Running**.
4. **Get the connection details** — on the cluster page click **Connect**:
   - Copy the **Public Endpoint** — this is your `ZILLIZ_URI`
     (format: `https://<cluster-id>.api.<region>.zillizcloud.com:443`).
   - Under **API Keys**, click **Create API Key**, give it a name, and copy the token.
     This is your `ZILLIZ_TOKEN`.
5. Paste both values into your `.env` file:

```env
ZILLIZ_URI=https://<cluster-id>.api.<region>.zillizcloud.com:443
ZILLIZ_TOKEN=<your-api-key>
```

> **Note:** The `video_traffic` Milvus collection is created automatically by `ingest.py` — you do not need to create it manually in the Zilliz console.

---

## Quickstart

### 1. Install dependencies

Make sure your venv is activated (see Prerequisites above), then:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY, ZILLIZ_URI, ZILLIZ_TOKEN
```

### 3. Ingest training data into Zilliz

```bash
python ingest.py
```

This will:
- Load every CSV under `dataset/train/`
- Extract and aggregate the `addr2_bytes` column to a 125-step timeseries
- Compute 19 statistical features per sample
- Fit a global z-score normaliser and save it to `normaliser.npz`
- Insert all vectors + metadata into the Zilliz collection `video_traffic`

### 4. Run the agent (folder watcher)

```bash
python agent.py
```

Then copy any CSV from `dataset/test/` into `watch_folder/`:

```bash
cp dataset/test/Netflix/vid3/Netflix_vid3_81.csv watch_folder/
```

The agent will automatically process the file and print a prediction, then delete it.

---

## How the agent works

The agent is built on the **OpenAI function-calling API** (GPT-4o). It acts as a
video traffic classification expert, using three internal tools to analyse a new
capture and produce a prediction. The entire reasoning pipeline is orchestrated
autonomously by the LLM — it decides when to call each tool and in what order.

### Agent tools

#### Tool 1 — `extract_timeseries`
Reads the CSV file, extracts the `addr2_bytes` column (downloaded bytes per
time window), and aggregates the raw 500-step series into a **125-step timeseries**
by summing every 4 consecutive values. Only the aggregated timeseries is returned
to the agent — the filepath and raw data are never exposed.

#### Tool 2 — `compute_stats_and_vector`
Takes the 125-step aggregated timeseries and computes **19 statistical features**
that characterise the traffic pattern:

| Feature | What it captures |
|---|---|
| mean, median | Central tendency of download volume |
| std, IQR, range | Spread / variability |
| min, max, P10, P90 | Extremes and tail behaviour |
| Q1, Q3 | Distribution quartiles |
| skewness | Asymmetry — e.g. bursty vs. gradual ramp |
| kurtosis | Peakedness — heavy-tailed vs. uniform flow |
| RMS | Power / energy intensity |
| CV (coeff. of variation) | Relative variability normalised by mean |
| energy | Sum of squares — total signal power |
| total | Overall downloaded bytes |
| zero_crossing_rate | How often the signal crosses its mean — burstiness indicator |
| agg_std_norm | Std relative to peak — normalised spread |

These are z-score normalised using the global mean/std fitted on the training set,
then returned as a fixed-length **feature vector** used for similarity search.

#### Tool 3 — `retrieve_similar`
Queries the **Zilliz vector database** with the feature vector using L2 distance
(IVF_FLAT index) and retrieves the **top-10 most similar training samples**. Each
result includes the platform label, video label, and the 125-step aggregated
timeseries of the matched sample — giving the agent concrete traffic patterns to
compare against.

### Reasoning and prediction

After running all three tools, the agent compares the query timeseries against the
retrieved patterns and reasons about:
- **Magnitude and energy** — is this a high or low bandwidth stream?
- **Burstiness** — sharp spikes vs. steady download flow
- **Distribution shape** — skewness, kurtosis, spread
- **Temporal structure** — where bursts and quiet periods occur

It then produces a final prediction in this format:

```
=== PREDICTION ===
Platform : Netflix
Video    : vid3
Reason   : The timeseries exhibits high-energy bursts concentrated in the first
           third of the window with a long low-activity tail, a strongly positive
           skew, and high kurtosis — a pattern characteristic of Netflix's
           adaptive bitrate delivery for this video's scene structure …
==================
```

> The agent presents its reasoning as expert knowledge about streaming platform
> traffic behaviour. It does not reveal that a database lookup was performed.

### Label isolation

To prevent the LLM from inferring the ground-truth label from the filename, the
watcher **renames every incoming file to a random UUID** before passing it to the
agent. The agent never sees the original filename.

---

## Data pipeline details

| Step | Description |
|------|-------------|
| Raw timeseries | 500 rows of `addr2_bytes` per CSV |
| Aggregation | Sum every 4 consecutive steps → 125-step timeseries |
| Features (19) | mean, std, min, max, median, Q1, Q3, IQR, skewness, kurtosis, RMS, CV, energy, total, range, P10, P90, zero-crossing-rate, normalised-std |
| Normalisation | Z-score using global mean/std fitted on the training set |
| Search metric | L2 distance, IVF_FLAT index, nprobe=16 |