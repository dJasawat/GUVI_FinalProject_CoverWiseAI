# CoverWise policy analyzer

This Python pipeline reads an insurance policy PDF, extracts page text, OCRs image
pages, removes common layout noise, creates section-aware retrieval chunks, and
asks Gemini or Groq for evidence-cited structured fields. The output includes
validated extraction data, source page metadata, and features suitable for a
downstream coverage-clarity or risk model. The features are not a policy
recommendation or a final risk score.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

OCR uses PyMuPDF's Tesseract integration. Install the Tesseract executable and
the needed language data separately, and ensure `tesseract.exe` is available on
`PATH`. The default OCR language is English (`eng`).

Set the provider API key. If only `GROQ_API_KEY` is set, Groq is selected
automatically. To select a provider explicitly, set `POLICY_LLM_PROVIDER`:

```powershell
$env:POLICY_LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "<your-key>"
```

For Groq, set `$env:GROQ_API_KEY = "<your-key>"`; optionally set
`$env:POLICY_LLM_PROVIDER = "groq"`. If both provider keys are configured, Gemini
remains the default unless `POLICY_LLM_PROVIDER` explicitly selects Groq. Models
can be overridden with `GEMINI_MODEL` or `GROQ_MODEL`. The default Gemini model is `gemini-2.5-flash`.

## Run the Streamlit app

Copy `.env.example` to `.env` and add your Gemini or Groq API key. Qdrant is
optional for chat over the uploaded PDFs; configure `QDRANT_URL` and
`QDRANT_API_KEY` only if you want the separate saved Q&A knowledge base.

```powershell
Copy-Item .env.example .env
# Edit .env and add your own key, then run:
.\.venv\Scripts\python.exe -m streamlit run appNew.py
```

Upload one or more PDFs from the sidebar. This repository's Streamlit entry
point is `appNew.py`. Uploaded policy PDFs are processed in the current Streamlit
session; temporary PDF files are removed after text extraction. Removing a policy
also clears its in-memory chunks and embeddings. Chat retrieval uses uploaded
policy chunks first; the separate Qdrant Q&A collection is optional and is skipped
if it is unavailable.

The **Summarize Policy** action extracts coverage, exclusions, waiting periods,
claims steps, renewal terms, and definitions with page-level evidence. The current
chat app's **Compare Policies** action is only a placeholder message; a complete
side-by-side comparison workflow is not implemented in this Streamlit screen yet.
Extracted results are informational and are not policy recommendations.

## Train the synthetic coverage-risk model

The repository includes a balanced 500-row synthetic dataset at
`Dataset/coverwise_synthetic_coverage_risk_dataset_500.csv`. It has 125 rows
for each of **Strong Coverage**, **Moderate Coverage**, **Caution Required**,
and **High Risk**. Its fields cover premiums, waiting periods, co-payment,
limits, exclusions, benefits, claim-process clarity, ambiguous clauses,
coverage clarity, and risk flags.

Regenerate the dataset deterministically, or create a different balanced size:

```powershell
.\.venv\Scripts\python.exe -m src.coverage_risk_model generate
.\.venv\Scripts\python.exe -m src.coverage_risk_model generate --rows-per-class 250 --seed 7
```

Compare logistic regression, a decision tree, and a random forest using a
stratified holdout set, then train and save the best macro-F1 classifier:

```powershell
.\.venv\Scripts\python.exe -m src.coverage_risk_model train
```

The command prints accuracy, macro F1, and the selected model's classification
report, and writes `model/coverage_risk_model.joblib`. The model can be loaded
and used from Python:

```python
from src.coverage_risk_model import load_model, predict_coverage_risk

model = load_model()
result = predict_coverage_risk(model, policy_features)
print(result["risk_label"], result["class_probabilities"])
```

The synthetic labels are generated from transparent feature rules, and
`coverage_clarity_score` is a synthetic feature rather than an extraction
result. Validation metrics measure agreement with those generated labels only;
they do not measure real-world underwriting performance. Extracted policy
PDF comparisons now request these features together with page-numbered quotes.
Every non-null extracted feature must have a quote that is checked against its
source page. Missing terms remain unknown; derived premium ratios, risk flags,
and clarity scores are calculated only from available inputs. The comparison
shows supporting quotes and marks predictions provisional when values must be
imputed from the synthetic training data. Results are educational indicators,
not real-world underwriting, financial, or insurance advice.

## Use as a Python function

```python
from src.policy_analyzer import analyze_policy

result = analyze_policy("Dataset/BajajHealthPolicy.pdf", provider="gemini")
print(result["extraction"])
print(result["scoring_features"])
print(result["ml_features"]["values"])
print(result["ml_features"]["evidence"])
```

The `ml_features` result contains the complete model input row plus supporting
page quotes. Values that are not present in the extracted policy remain `None`;
`missing_features` lists them. The Policy Comparison tab displays these inputs
side by side and runs the trained model, marking a result provisional when it
had to impute unknown values. If every model feature is unknown, it does not
produce a risk class.

`analyze_policy` also accepts PDF bytes or a binary file object, so an upload
handler can pass the uploaded content directly. PDF inputs are capped at 40 MB.
Every extracted clause must include a page number and a quote that is verified
against the extracted page text.

## Run from the command line

```powershell
.\.venv\Scripts\python.exe -m src.policy_analyzer Dataset\BajajHealthPolicy.pdf --provider groq --output analysis.json
```

The JSON includes the policy metadata, extracted fields, scoring features, and
all chunks with policy name, page number(s), clause heading, and section title.
The retrieval stage lexically selects the most relevant chunks before the LLM
request; it does not persist them to a vector database.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
