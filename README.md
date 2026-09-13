# News Bias Detector

A single Jupyter/Colab notebook (`NewsBiasDetector.ipynb`) that scrapes news headlines, cleans them, and trains a series of increasingly complex ML models to classify which outlet a headline came from — Fox News or NBC News — as a proxy for detecting outlet-level bias in language.

> Built as a class project (CIS 5190). The notebook is a linear, exploratory pipeline rather than a packaged library — run it top to bottom in Colab.

## What it does

The notebook walks through five stages, laid out with markdown headers you can navigate directly in Colab/Jupyter:

1. **Data Collection** — scrapes article headlines from a list of URLs (`df_train_url`) using `requests` + `BeautifulSoup`, falling back from the page's `<h1>` to its `og:title` meta tag, with retry/backoff logic. Output: `scraped_headlines.csv`.
2. **Data Preprocessing** — lowercases text, strips punctuation, removes stopwords, and lemmatizes tokens (NLTK). Deduplicates headlines and derives a `Source` label from the article's domain (`foxnews` / `nbcnews`). Output: `preprocessed_headlines.csv`.
3. **Model Implementation** — a TF-IDF + Logistic Regression baseline, then a bake-off across nine classifiers (Naive Bayes, Logistic Regression, Linear SVM, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Bagging, Radius Neighbors, XGBoost, KNN, MLP) to shortlist the strongest approaches.
4. **Model Fine-Tuning** — grid/random search hyperparameter tuning for Logistic Regression and Linear SVM (TF-IDF pipelines); a CNN and an RNN (LSTM) built in TensorFlow/Keras with `keras-tuner` search; a fine-tuned small BERT model (`small_bert/bert_en_uncased_L-4_H-512_A-8`) via TensorFlow Hub/TF-Models, with its own grid and random hyperparameter search; and finally a **VotingClassifier ensemble** built on TF-IDF features plus hand-engineered text features (headline length, punctuation counts, etc.).
5. **Model Evaluation** — accuracy, precision/recall/F1, and confusion matrices for each model, plus a final scoring pass of the tuned model against a held-out `preprocessed_new_data.csv` file, written out to `submission_new_test.csv`.

## Results

Test-set accuracy (binary Fox News vs. NBC News headline classification), from the notebook's recorded runs:

| Model | Accuracy |
|---|---|
| Radius Neighbors | 52.8% |
| AdaBoost | 62.6% |
| Decision Tree | 70.8% |
| Gradient Boosting | 74.5% |
| Bagging | 75.7% |
| XGBoost | 77.3% |
| Random Forest | 77.5% |
| KNN | 77.9% |
| Naive Bayes | 79.9% |
| MLP | 79.9% |
| Ensemble (engineered features) | 80.7% |
| Logistic Regression (baseline) | 81.1% |
| BERT (small, fine-tuned, tuned) | ~81.0% |
| SVM (LinearSVC) | 81.9% |
| **Logistic Regression (tuned, TF-IDF grid search)** | **82.8%** |

The tuned TF-IDF + Logistic Regression pipeline was the best overall performer; the CNN/RNN deep learning models underperformed the classical baselines on this dataset size.

## Requirements

The notebook is written for **Google Colab** (it opens with a Google Drive mount) but can be adapted to run locally. Core dependencies used across the notebook:

- `pandas`, `numpy`, `matplotlib`
- `requests`, `beautifulsoup4`
- `nltk` (`stopwords`, `wordnet`, `omw-1.4` corpora)
- `scikit-learn`
- `xgboost`
- `tensorflow`, `tensorflow-text`, `tensorflow-hub`, `tf-models-official`, `tensorflow-datasets`, `tf-keras`
- `keras-tuner`
- `scipy`

Install locally with:

```bash
pip install pandas numpy matplotlib requests beautifulsoup4 nltk scikit-learn xgboost \
            tensorflow tensorflow-text tensorflow-hub tf-models-official \
            tensorflow-datasets tf-keras keras-tuner scipy
```

## Usage

1. Open `NewsBiasDetector.ipynb` in Google Colab (or Jupyter, after removing/adjusting the `google.colab.drive.mount(...)` cell and the working-directory `os.chdir(...)` call).
2. Provide a CSV of source article URLs as `df_train_url` for the scraping step, or skip straight to preprocessing if you already have `scraped_headlines.csv`.
3. Run cells top to bottom, section by section — later sections (SVM tuning, CNN/RNN, BERT, ensemble) each assume the TF-IDF/train-test split objects from earlier cells are still in memory.
4. To score new headlines, place them (pre-cleaned, in a `Cleaned_Headline` column) in `preprocessed_new_data.csv`; the last cell will run the tuned model against it and write predictions to `submission_new_test.csv`.

## Notes & caveats

- The scraper is best-effort: it depends on each site's `<h1>` / `og:title` markup and includes randomized delays and retries to be polite to target servers. Web scraping is subject to each source's terms of service — review them before re-running collection against live sites.
- "Bias" here is operationalized narrowly as *outlet classification from headline text* (can a model tell a Fox News headline from an NBC News headline?), not a general-purpose political bias or fact-checking score. Results should be read with that scope in mind.
- The BERT and CNN/RNN cells are GPU-friendly but can be slow on CPU; hyperparameter search cells (`GridSearchCV`, `RandomizedSearchCV`, `keras-tuner`) are the most time-consuming and were run with reduced trial counts in the original notebook.
- Some intermediate CSVs (`scraped_headlines.csv`, `preprocessed_headlines.csv`, `preprocessed_new_data.csv`) are expected as inputs/outputs of specific cells but are not included in this repo — you'll need to supply your own source URLs or data to reproduce the full pipeline end to end.
