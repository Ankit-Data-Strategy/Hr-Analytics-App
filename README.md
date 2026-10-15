# HR Attrition Predictor & Consulting Assistant

Upload a raw HR dataset → clean it automatically → predict attrition risk per
employee → answer a few questions about company context → get an
AI-generated, data-backed retention strategy report.

## What's inside

```
hr_attrition_app/
├── app.py                  # Streamlit application (run this)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Theme colors for Streamlit's native widgets
├── models/
│   └── attrition_model.pkl # Trained model (already built for you)
├── data/
│   └── Cleaned_data.csv    # Training data used to build the model
└── src/
    ├── clean_data.py       # Refactored version of your cleaning notebook
    ├── train_model.py      # Trains the model (already run once — see below)
    ├── predict.py           # Cleans + scores a fresh upload
    ├── prompt_builder.py    # Builds the prompt sent to Gemini
    ├── gemini_api.py         # Calls the Gemini API
    └── ui_style.py           # Design tokens, cards, step tracker, risk gauge
```

## Design

The UI uses a "workforce risk dashboard" visual language rather than default
Streamlit styling:

- **Colors**: deep navy header/sidebar, light neutral background, cards on
  white, and semantic risk colors (green/amber/red) used consistently for
  risk badges, the department chart, and the gauge — so color always means
  the same thing everywhere in the app.
- **Type**: Space Grotesk for headings, Inter for body text, IBM Plex Mono
  for all numbers/metrics (a deliberate "data tool" signature).
- **Signature element**: a custom SVG radial gauge showing the overall
  Workforce Risk Score, generated in `ui_style.risk_gauge_svg()` — no charting
  library dependency for it, pure SVG.
- **Step tracker**: the sidebar shows your progress through the 4-step flow
  instead of being just a settings panel.

To restyle it, everything lives in one place: the `COLORS` dict at the top of
`src/ui_style.py`. Change a hex value there and it propagates everywhere
(badges, gauge, chart bars, headers).

Fonts load from Google Fonts via CSS `@import`, so an internet connection is
needed for them to render correctly (they'll fall back to system fonts
offline, which still looks fine, just less distinctive).

## How the model was built

Your original pipeline used Orange with:
- **Logistic Regression** (C=1.0, L2 penalty — Orange's defaults)
- Wrapped in a **Calibrated Learner** (Sigmoid/Platt calibration)
- With the decision **threshold optimized for F1 score** instead of the default 0.5

`src/train_model.py` replicates this exactly in scikit-learn:
`LogisticRegression(C=1.0, penalty='l2')` → `CalibratedClassifierCV(method='sigmoid')`
→ threshold selected by scanning for max F1 on the training data.

The cleaning logic in `src/clean_data.py` was verified to reproduce your
`Cleaned_data.csv` **exactly**, row for row, from the raw messy export.

The model has already been trained and saved to `models/attrition_model.pkl`
— you don't need to retrain it unless you get new training data. If you ever
do want to retrain (e.g. with more historical data), just replace
`data/Cleaned_data.csv` and run:

```bash
python src/train_model.py
```

**Current model performance** (on a held-out 20% test split):
- AUC: 0.69
- F1: 0.33 (Precision 0.25, Recall 0.51)
- Attrition base rate: ~13%

This is a modest but real signal — logistic regression with default settings.
It correctly catches about half of actual leavers, at the cost of some false
positives, which is a reasonable tradeoff for a retention *early-warning*
tool where the cost of missing a leaver is usually higher than the cost of
double-checking on someone who was fine after all.

## Setup

1. **Install dependencies:**
   ```bash
   cd hr_attrition_app
   pip install -r requirements.txt
   ```

2. **Get a Google Gemini API key** (if you don't already have one):
   https://aistudio.google.com/apikey

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   This opens in your browser at `http://localhost:8501`.

4. **In the app:**
   - Paste your Gemini API key in the sidebar (only used for this session, never saved)
   - Upload a CSV of employees in the same raw format as your training data
     (columns like `Title`, `BusinessUnit`, `DepartmentType`, `EngagementScore`,
     `SatisfactionScore`, `StartDate`, `DOB`, etc. — see `data/Cleaned_data.csv`
     columns for the target schema, or check `src/clean_data.py`'s `FEATURE_COLUMNS`)
   - Click "Run attrition predictions"
   - Fill in the company context form (policy changes, budget constraints, etc.)
   - Click "Generate AI recommendations" for the full consulting report

## Expected input format

The uploaded CSV should match the shape of `Messy_HR_Dataset_Capstone.csv` —
i.e. it should NOT need to already have an `Attrition` or outcome column
(that's what you're trying to predict). It DOES need an `EmployeeStatus`-style
column removed/absent, plus the raw fields the cleaner expects:
`StartDate`, `DOB`, `Title`, `BusinessUnit`, `EmployeeType`, `PayZone`,
`EmployeeClassificationType`, `DepartmentType`, `Division`, `State`,
`GenderCode`, `RaceDesc`, `MaritalDesc`, `Performance Score`,
`Current Employee Rating`, `Engagement Score`, `Satisfaction Score`,
`Work-Life Balance Score`, `Training Program Name`, `Training Type`,
`Training Outcome`, `Training Duration(Days)`, `Training Cost`.

If your new upload has a different column naming convention (e.g. no spaces
already), you may need to adjust `RENAME_MAP` in `src/clean_data.py`.

## Notes / things to know

- **No data leaves your machine except what's sent to Gemini.** The prompt
  sent to Gemini includes aggregate stats and the top 10 highest-risk
  employees' *feature data* (department, scores, tenure, etc.) — but never
  names, emails, or employee IDs, even if you choose to display names in
  the app's results table.
- **API costs**: each "Generate AI recommendations" click makes one Gemini
  API call (~2000 output tokens). Check current pricing at
  https://ai.google.dev/gemini-api/docs/pricing if you're running this frequently or at scale.
- **Retraining**: if you get much more training data in the future, rerun
  `train_model.py` — no code changes needed elsewhere.
