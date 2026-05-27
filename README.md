# TuneAI - Automated ML Model Comparison Suite

TuneAI came out of a project during my doctoral period where I needed to compare multiple different classifiers on the same dataset, tune each one properly, and then see a clear picture of which model actually performed best. Doing that by hand - writing the same grid search loop for every algorithm, fixing deprecated API calls, wrestling with plot windows - got old fast. So I built this tool to do it automatically.

Any training data in xlsx format can be used that has features in the columns and a binary label in the last column, and TuneAI handles the rest: it scales the data, runs a hyperparameter search for every classifier, evaluates each one on a held-out test split, and presents a side-by-side performance dashboard at the end.

---

## What it does

When you run the tool it opens a file picker. Select your Excel file and walk away - the software will:

1. Load and describe your dataset (sample count, class balance, feature count)
2. Scale features using Min-Max normalisation and split 70/30 into train and test sets
3. Train and tune seven classifiers in sequence:
   - **ANN** - three-layer neural network, tuned with Keras Tuner (Hyperband)
   - **SVM** - Support Vector Machine with an RBF kernel
   - **XGBoost** - Extreme Gradient Boosting
   - **Random Forest** - ensemble of decision trees
   - **Logistic Regression** - regularised linear classifier
   - **Naive Bayes** - Gaussian probabilistic classifier
   - **KNN** - K-Nearest Neighbours
4. Show a confusion matrix after each model runs
5. Print a formatted comparison table (accuracy, precision, recall, F1-score)
6. Render a grouped bar chart dashboard so you can see everything at a glance

For the scikit-learn models, both GridSearchCV and RandomizedSearchCV are applied and the better result is kept.  The ANN uses Keras Tuner's Hyperband algorithm, which is fast and thorough.

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ShadAhammed/TuneAI.git
cd TuneAI
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Prepare your data

Your Excel file needs to follow a simple structure:

| Index | Feature 1 | Feature 2 | ... | Target |
|-------|-----------|-----------|-----|--------|
| 1     | 0.42      | 1.7       | ... | 1      |
| 2     | 0.91      | 0.3       | ... | 0      |

- The first column is treated as the index.
- Every column except the last is treated as a feature.
- The last column must be a binary label (0 or 1).
- The column names can be anything - TuneAI does not rely on specific names.

> **Privacy note:** The `data/` folder is excluded from version control. Never commit real research or patient data to the repository.

### 4. Run the tool

```bash
python run.py
```

A file picker will open. Select your `.xlsx` file and watch the models run.

---

## Running multiple datasets in batch mode

If you want to evaluate several files without going through the dialog each time, edit `models/main.py` and add your file paths to the `DATASETS` list:

```python
DATASETS = [
    (r'C:\path\to\your\data\cohort_a.xlsx', 'Cohort A'),
    (r'C:\path\to\your\data\cohort_b.xlsx', 'Cohort B'),
]
```

Then run:

```bash
python models/main.py
```

---

## Project structure

```
TuneAI/
├── run.py                      # Main entry point - run this
├── requirements.txt
├── models/
│   ├── main.py                 # Batch runner for multiple datasets
│   ├── MLModels.py             # ModelRunner - orchestrates all classifiers
│   └── ModelParams.py          # Hyperparameter grids for each classifier
├── src/
│   ├── DataExp/
│   │   ├── DataSource.py       # File-picker dialog helper
│   │   └── TrgData.py          # Data scaling, splitting, and statistics
│   ├── models/
│   │   ├── ANN.py              # Neural network with Keras Tuner
│   │   ├── SVM.py              # Support Vector Machine
│   │   ├── XGB.py              # XGBoost
│   │   ├── RF.py               # Random Forest
│   │   ├── LR.py               # Logistic Regression
│   │   ├── NB.py               # Naive Bayes
│   │   ├── KNN.py              # K-Nearest Neighbours
│   │   └── ModelTuning.py      # Shared GridSearch / RandomSearch wrapper
│   └── visualization/
│       └── Performance.py      # Confusion matrices and dashboard chart
└── data/                       # NOT in git - put your data files here
```

---

## Dependencies

The main libraries are listed in `requirements.txt`.  Key ones:

| Library | Purpose |
|---------|---------|
| TensorFlow 2.13 | ANN training |
| Keras Tuner 1.4 | ANN hyperparameter search |
| scikit-learn 1.3 | All other classifiers + search |
| XGBoost 1.7 | Gradient boosting |
| pandas 2.0 | Data loading and manipulation |
| matplotlib 3.7 | Confusion matrices and dashboard |
| openpyxl | Reading `.xlsx` files |
| tabulate | Formatted summary table |

---

## Adapting TuneAI to your own project

The tool is intentionally generic. A few things worth knowing:

- **Target column:** TuneAI always treats the last column of your spreadsheet as the label. No configuration needed.
- **Adding a model:** Create a new class in `src/models/` following the same pattern as `LR.py`, add its parameters to `models/ModelParams.py`, and call it from `ModelRunner.RunModel()` in `models/MLModels.py`.
- **Changing the search strategy:** Edit the `Tuner` class in `src/models/ModelTuning.py` or adjust the ANN search in `src/models/ANN.py`.
- **Tuning the ANN search:** Change `tuner_id` in `ModelRunner.RunModel()` - `1` = RandomSearch, `2` = Hyperband (default), `3` = BayesianOptimization.

---

## Author

Abu Shad Ahammed  
Chair of Embedded Systems, Universität Siegen

---

## License

BSD 3-Clause - see [LICENSE](LICENSE) for full terms.
