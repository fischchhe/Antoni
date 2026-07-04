# The Antoni Corpus — Pipeline

Code for building and analysing a multilingual parallel corpus of Antoni van
Leeuwenhoek's letters (17th/18th c.) and their English translations originally
my B.A. thesis (*A Linguistic Investigation of Historical Scientific Texts*,
University of Mannheim, 2025), now being extended at the
University of Stuttgart.

## Research question

Did the editorial process of publishing Leeuwenhoek's letters in the Royal
Society's *Philosophical Transactions* strip out his first-person,
self-referential voice, compared to the Dutch originals and other English
translations?

## The corpus (≈758,000 tokens, four parallel subcorpora)

| Subcorpus | Description |
|---|---|
| `Dutch` | Leeuwenhoek's original Dutch letters |
| `PT` | Contemporary *Philosophical Transactions* versions (OCR'd from page scans) |
| `Letterbook` | Contemporary Letterbook translations |
| `Allebrieven` | Modern 20th-century English translations (*Alle de Brieven*) |

**Note: the corpus texts themselves are not included in this repository** — the
source material belongs to the Royal Society archives and the *Alle de Brieven*
edition and is not mine to redistribute. This repo contains the pipeline code
only.

## Pipeline

Scripts are numbered in the order the data flows:

1. **`01_scrape.py`** — download letter PDFs from the Royal Society's archive
   (reference numbers from an Excel sheet).
2. **`02_ocr_gvision.py`** — OCR the *Philosophical Transactions* page scans
   with Google Cloud Vision (document text detection), stitch pages per letter.
3. **`03_tokenize_postag.py`** — tokenize and POS-tag all four subcorpora with
   spaCy (`nl_core_news_md` / `en_core_web_sm`).
4. **`04_frequency_count.py`** — count first-person pronouns per letter,
   normalized per 1,000 tokens. The pronoun lists include early modern spelling
   variants and systematic OCR artefacts (long-s forms like *myſelf*, f-for-s
   misreads like *myfelf*, v-for-u like *vs*) found by inspecting the OCR
   output. Formulaic openings/closings (first & last 3 paragraphs) are excluded
   for the letter subcorpora.
5. **`05_stats.py`** — one-way ANOVA across subcorpora with assumption checks
   (Shapiro-Wilk on residuals and per group, Levene), robust (HC3) and Welch's
   ANOVA, plots.
6. **`06_stats_log.py`** — the same analysis on log-transformed frequencies
   (normality of residuals failed on the raw scale), plus planned Welch t-tests
   (PT vs. Allebrieven, PT vs. Letterbook) with Bonferroni-corrected α = .025,
   Cohen's d, and an APA-style results table.
7. **`07_align_single_letter.py`** — prototype: GPT-4o sentence alignment of the
   four versions of one letter into JSON.
8. **`08_align_all.py`** — the same alignment batched over the whole corpus,
   producing a sentence-aligned parallel corpus for qualitative analysis.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download nl_core_news_md
```

Credentials go in environment variables (see `.env.example`):
`OPENAI_API_KEY` for the alignment scripts, `GOOGLE_APPLICATION_CREDENTIALS`
for the OCR step.

## Status & next steps

- [x] Corpus construction, frequency analysis, statistical testing (B.A. thesis, grade 1.3)
- [ ] Refactor into a documented Python library
- [ ] RAG system for diachronic-linguistic queries over the corpus (proposal stage)

## Author

Christiane Henrike Fischer — M.Sc. Computational Linguistics, University of
Stuttgart.
